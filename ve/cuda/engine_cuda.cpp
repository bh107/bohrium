/*
This file is part of Bohrium and copyright (c) 2012 the Bohrium
team <http://www.bh107.org>.

Bohrium is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3
of the License, or (at your option) any later version.

Bohrium is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the
GNU Lesser General Public License along with Bohrium.

If not, see <http://www.gnu.org/licenses/>.
*/

#include <vector>
#include <iostream>
#include <boost/functional/hash.hpp>
#include <boost/algorithm/string/replace.hpp>
#include <iomanip>

#include "engine_cuda.hpp"

using namespace std;
namespace fs = boost::filesystem;

namespace bohrium {

static boost::hash<string> hasher;

EngineCUDA::EngineCUDA(const ConfigParser &config, jitk::Statistics &stat) :
                                    work_group_size_1dx(config.defaultGet<int>("work_group_size_1dx", 128)),
                                    work_group_size_2dx(config.defaultGet<int>("work_group_size_2dx", 32)),
                                    work_group_size_2dy(config.defaultGet<int>("work_group_size_2dy", 4)),
                                    work_group_size_3dx(config.defaultGet<int>("work_group_size_3dx", 32)),
                                    work_group_size_3dy(config.defaultGet<int>("work_group_size_3dy", 2)),
                                    work_group_size_3dz(config.defaultGet<int>("work_group_size_3dz", 2)),
                                    compile_flg(config.defaultGet<string>("compiler_flg", "")),
                                    default_device_type(config.defaultGet<string>("device_type", "auto")),
                                    platform_no(config.defaultGet<int>("platform_no", -1)),
                                    verbose(config.defaultGet<bool>("verbose", false)),
                                    cache_file_max(config.defaultGet<int64_t>("cache_file_max", 50000)),
                                    stat(stat),
                                    prof(config.defaultGet<bool>("prof", false)),
                                    tmp_dir(jitk::get_tmp_path(config)),
                                    tmp_src_dir(tmp_dir / "src"),
                                    tmp_bin_dir(tmp_dir / "obj"),
                                    cache_bin_dir(fs::path(config.defaultGet<string>("cache_dir", "")))
{
    int deviceCount = 0;
    CUresult err = cuInit(0);

    if (err == CUDA_SUCCESS)
        checkCudaErrors(cuDeviceGetCount(&deviceCount));

    if (deviceCount == 0) {
        throw runtime_error("Error: no devices supporting CUDA");
    }

    // get first CUDA device
    checkCudaErrors(cuDeviceGet(&device, 0));

    err = cuCtxCreate(&context, 0, device);
    if (err != CUDA_SUCCESS) {
        cuCtxDetach(context);
        throw runtime_error("Error initializing the CUDA context.");
    }

    // Let's make sure that the directories exist
    jitk::create_directories(tmp_src_dir);
    jitk::create_directories(tmp_bin_dir);
    if (not cache_bin_dir.empty()) {
        jitk::create_directories(cache_bin_dir);
    }

    // Write the compilation hash
    compilation_hash = hasher(info());

    // Get the compiler command and replace {MAJOR} and {MINOR} with the SM versions
    string compiler_cmd = config.get<string>("compiler_cmd");
    {
        int major = 0, minor = 0;
        checkCudaErrors(cuDeviceComputeCapability(&major, &minor, device));
        boost::replace_all(compiler_cmd, "{MAJOR}", std::to_string(major));
        boost::replace_all(compiler_cmd, "{MINOR}", std::to_string(minor));
    }

    // Init the compiler
    compiler = jitk::Compiler(compiler_cmd, verbose);
}

EngineCUDA::~EngineCUDA() {
    cuCtxDetach(context);

    // Move JIT kernels to the cache dir
    if (not cache_bin_dir.empty()) {
        for (const auto &kernel: _functions) {
            const fs::path src = tmp_bin_dir / jitk::hash_filename(compilation_hash, kernel.first, ".cubin");
            if (fs::exists(src)) {
                const fs::path dst = cache_bin_dir / jitk::hash_filename(compilation_hash, kernel.first, ".cubin");
                if (not fs::exists(dst)) {
                    fs::copy_file(src, dst);
                }
            }
        }
    }

    // File clean up
    if (not verbose) {
        fs::remove_all(tmp_src_dir);
    }

    if (cache_file_max != -1) {
        util::remove_old_files(cache_bin_dir, cache_file_max);
    }
}

pair<tuple<uint32_t, uint32_t, uint32_t>, tuple<uint32_t, uint32_t, uint32_t> > EngineCUDA::NDRanges(const vector<const jitk::LoopB*> &threaded_blocks) const {
    const auto &b = threaded_blocks;
    switch (b.size()) {
        case 1: {
            const auto gsize_and_lsize = jitk::work_ranges(work_group_size_1dx, b[0]->size);
            return make_pair(make_tuple(gsize_and_lsize.first, 1, 1), make_tuple(gsize_and_lsize.second, 1, 1));
        }
        case 2: {
            const auto gsize_and_lsize_x = jitk::work_ranges(work_group_size_2dx, b[0]->size);
            const auto gsize_and_lsize_y = jitk::work_ranges(work_group_size_2dy, b[1]->size);
            return make_pair(make_tuple(gsize_and_lsize_x.first, gsize_and_lsize_y.first, 1),
                             make_tuple(gsize_and_lsize_x.second, gsize_and_lsize_y.second, 1));
        }
        case 3: {
            const auto gsize_and_lsize_x = jitk::work_ranges(work_group_size_3dx, b[0]->size);
            const auto gsize_and_lsize_y = jitk::work_ranges(work_group_size_3dy, b[1]->size);
            const auto gsize_and_lsize_z = jitk::work_ranges(work_group_size_3dz, b[2]->size);
            return make_pair(make_tuple(gsize_and_lsize_x.first, gsize_and_lsize_y.first, gsize_and_lsize_z.first),
                             make_tuple(gsize_and_lsize_x.second, gsize_and_lsize_y.second, gsize_and_lsize_z.second));
        }
        default:
            throw runtime_error("NDRanges: maximum of three dimensions!");
    }
}

CUfunction EngineCUDA::getFunction(const string &source) {
    size_t hash = hasher(source);
    ++stat.kernel_cache_lookups;

    // Do we have the program already?
    if (_functions.find(hash) != _functions.end()) {
        return _functions.at(hash);
    }

    fs::path binfile = cache_bin_dir / jitk::hash_filename(compilation_hash, hash, ".cubin");

    // If the binary file of the kernel doesn't exist we create it
    if (verbose or cache_bin_dir.empty() or not fs::exists(binfile)) {
        ++stat.kernel_cache_misses;

        // We create the binary file in the tmp dir
        binfile = tmp_bin_dir / jitk::hash_filename(compilation_hash, hash, ".cubin");

        // Write the source file and compile it (reading from disk)
        // TODO: make nvcc read directly from stdin
        {
            fs::path srcfile = jitk::write_source2file(source, tmp_src_dir,
                                                       jitk::hash_filename(compilation_hash, hash, ".cu"), verbose);
            compiler.compile(binfile.string(), srcfile.string());
        }
        /* else {
            // Pipe the source directly into the compiler thus no source file is written
            compiler.compile(binfile.string(), source.c_str(), source.size());
        }
       */
    }

    CUmodule module;
    CUresult err = cuModuleLoad(&module, binfile.string().c_str());
    if (err != CUDA_SUCCESS) {
        const char *err_name, *err_desc;
        cuGetErrorName(err, &err_name);
        cuGetErrorString(err, &err_desc);
        cout << "Error loading the module \"" << binfile.string()
             << "\", " << err_name << ": \"" << err_desc << "\"." << endl;
        cuCtxDetach(context);
        throw runtime_error("cuModuleLoad() failed");
    }

    CUfunction program;
    err = cuModuleGetFunction(&program, module, "execute");
    if (err != CUDA_SUCCESS) {
        const char *err_name, *err_desc;
        cuGetErrorName(err, &err_name);
        cuGetErrorString(err, &err_desc);
        cout << "Error getting kernel function 'execute' \"" << binfile.string()
             << "\", " << err_name << ": \"" << err_desc << "\"." << endl;
        throw runtime_error("cuModuleGetFunction() failed");
    }
    _functions[hash] = program;
    return program;
}

void EngineCUDA::execute(const std::string &source, const std::vector<bh_base*> &non_temps,
                           const vector<const jitk::LoopB*> &threaded_blocks,
                           const vector<const bh_view*> &offset_strides,
                           const vector<const bh_instruction*> &constants) {

    auto tcompile = chrono::steady_clock::now();
    CUfunction program = getFunction(source);
    stat.time_compile += chrono::steady_clock::now() - tcompile;

    // Let's execute the CUDA kernel
    vector<void *> args;

    for (bh_base *base: non_temps) { // NB: the iteration order matters!
        args.push_back(getBuffer(base));
    }

    for (const bh_view *view: offset_strides) {
        args.push_back((void*)&view->start);
        for (int j=0; j<view->ndim; ++j) {
            args.push_back((void*)&view->stride[j]);
        }
    }

    auto texec = chrono::steady_clock::now();

    tuple<uint32_t, uint32_t, uint32_t> blocks, threads;
    tie(blocks, threads) = NDRanges(threaded_blocks);

    checkCudaErrors(cuLaunchKernel(program,
                                   get<0>(blocks), get<1>(blocks), get<2>(blocks),  // NxNxN blocks
                                   get<0>(threads), get<1>(threads), get<2>(threads),  // NxNxN threads
                                   0, 0, &args[0], 0));
    checkCudaErrors(cuCtxSynchronize());

    stat.time_exec += chrono::steady_clock::now() - texec;
}

void EngineCUDA::set_constructor_flag(std::vector<bh_instruction*> &instr_list) {
    jitk::util_set_constructor_flag(instr_list, buffers);
}

std::string EngineCUDA::info() const {

    char device_name[1000];
    cuDeviceGetName(device_name, 1000, device);
    int major = 0, minor = 0;
    checkCudaErrors(cuDeviceComputeCapability(&major, &minor, device));
    size_t totalGlobalMem;
    checkCudaErrors(cuDeviceTotalMem(&totalGlobalMem, device));

    stringstream ss;
    ss << "----"                                                                        << "\n";
    ss << "CUDA:"                                                                       << "\n";
    ss << "  Device: \"" << device_name << " (SM " << major << "." << minor << " compute capability)\"\n";
    ss << "  Memory: \"" <<totalGlobalMem / 1024 / 1024 << " MB\"\n";
    ss << "  JIT Command: \"" << compiler.cmd_template << "\"\n";
    return ss.str();
}

} // bohrium
