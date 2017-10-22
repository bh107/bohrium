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
#include <boost/filesystem.hpp>

#include "engine_opencl.hpp"

namespace fs = boost::filesystem;
using namespace std;

namespace {

#define CL_DEVICE_AUTO 1024 // More than maximum in the bitmask

map<const string, cl_device_type> device_map = {
    {"auto",        CL_DEVICE_AUTO},
    {"gpu",         CL_DEVICE_TYPE_GPU},
    {"accelerator", CL_DEVICE_TYPE_ACCELERATOR},
    {"default",     CL_DEVICE_TYPE_DEFAULT},
    {"cpu",         CL_DEVICE_TYPE_CPU}
};

// Get the OpenCL device (search order: GPU, ACCELERATOR, DEFAULT, and CPU)
cl::Device getDevice(const cl::Platform &platform, const string &default_device_type) {
    vector<cl::Device> device_list;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &device_list);

    if(device_list.empty()){
        throw runtime_error("No OpenCL device found");
    }

    if (!util::exist(device_map, default_device_type)) {
        stringstream ss;
        ss << "'" << default_device_type << "' is not a OpenCL device type. " \
           << "Must be one of 'auto', 'gpu', 'accelerator', 'cpu', or 'default'";
        throw runtime_error(ss.str());
    } else if (device_map[default_device_type] != CL_DEVICE_AUTO) {
        for (auto &device: device_list) {
            if ((device.getInfo<CL_DEVICE_TYPE>() & device_map[default_device_type]) == device_map[default_device_type]) {
                return device;
            }
        }
        stringstream ss;
        ss << "Could not find selected OpenCL device type ('" \
           << default_device_type << "') on default platform";
        throw runtime_error(ss.str());
    }

    // Type was 'auto'
    for (auto &device_type: device_map) {
        for (auto &device: device_list) {
            if ((device.getInfo<CL_DEVICE_TYPE>() & device_type.second) == device_type.second) {
                return device;
            }
        }
    }
    throw runtime_error("No OpenCL device of usable type found");
}
}

namespace bohrium {

static boost::hash<string> hasher;

EngineOpenCL::EngineOpenCL(const ConfigParser &config, jitk::Statistics &stat) :
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
                                    stat(stat),
                                    prof(config.defaultGet<bool>("prof", false)),
                                    tmp_dir(jitk::get_tmp_path(config)),
                                    tmp_src_dir(tmp_dir / "src"),
                                    tmp_bin_dir(tmp_dir / "obj"),
                                    cache_bin_dir(fs::path(config.defaultGet<string>("cache_dir", "")))
{
    vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if(platforms.empty()) {
        throw runtime_error("No OpenCL platforms found");
    }

    bool found = false;
    if (platform_no == -1) {
        for (auto pform : platforms) {
            // Pick first valid platform
            try {
                // Get the device of the platform
                platform = pform;
                device = getDevice(platform, default_device_type);
                found = true;
            } catch(cl::Error err) {
                // We try next platform
            }
        }
    } else {
        if (platform_no > ((int) platforms.size()-1)) {
            std::stringstream ss;
            ss << "No such OpenCL platform. Tried to fetch #";
            ss << platform_no << " out of ";
            ss << platforms.size()-1 << "." << endl;
            throw std::runtime_error(ss.str());
        }

        platform = platforms[platform_no];
        device = getDevice(platform, default_device_type);
        found = true;
    }

    if (verbose) {
        cout << "Using platform: " << platform.getInfo<CL_PLATFORM_NAME>() << endl;
    }

    if (!found) {
        throw runtime_error("Invalid OpenCL device/platform");
    }

    if(verbose) {
        cout << "Using device: " << device.getInfo<CL_DEVICE_NAME>() \
             << " ("<< device.getInfo<CL_DEVICE_OPENCL_C_VERSION>() << ")" << endl;
    }

    context = cl::Context(device);
    queue = cl::CommandQueue(context, device);

    // Let's make sure that the directories exist
    jitk::create_directories(tmp_src_dir);

    // Write the compilation hash
    stringstream ss;
    ss << compile_flg
       << platform.getInfo<CL_PLATFORM_NAME>()
       << device.getInfo<CL_DEVICE_NAME>()
       << device.getInfo<CL_DEVICE_OPENCL_C_VERSION>();
    compilation_hash = hasher(ss.str());
}

EngineOpenCL::~EngineOpenCL() {
    // Move JIT kernels to the cache dir
    if (not cache_bin_dir.empty()) {
        for (const auto &kernel: _programs) {
            const fs::path dst = cache_bin_dir / jitk::hash_filename(compilation_hash, kernel.first, ".clbin");
            if (not fs::exists(dst)) {
                cl_uint ndevs;
                kernel.second.getInfo(CL_PROGRAM_NUM_DEVICES, &ndevs);
                if (ndevs > 1) {
                    cout << "OpenCL warning: too many devices for caching." << endl;
                    return;
                }
                size_t bin_sizes[1];
                kernel.second.getInfo(CL_PROGRAM_BINARY_SIZES, bin_sizes);
                if (bin_sizes[0] == 0) {
                    cout << "OpenCL warning: no caching since the binary isn't available for the device." << endl;
                } else {
                    // Get the CL_PROGRAM_BINARIES and write it to a file
                    vector<unsigned char> bin(bin_sizes[0]);
                    unsigned char *bin_list[1] = {&bin[0]};
                    kernel.second.getInfo(CL_PROGRAM_BINARIES, bin_list);
                    ofstream binfile(dst.string(), ofstream::out | ofstream::binary);
                    binfile.write((const char*)&bin[0], bin.size());
                    binfile.close();
                }
            }
        }
    }

    // File clean up
    if (not verbose) {
        fs::remove_all(tmp_src_dir);
    }
}

pair<cl::NDRange, cl::NDRange> EngineOpenCL::NDRanges(const vector<const jitk::LoopB*> &threaded_blocks) const {
    const auto &b = threaded_blocks;
    switch (b.size()) {
        case 1: {
            const auto gsize_and_lsize = jitk::work_ranges(work_group_size_1dx, b[0]->size);
            return make_pair(cl::NDRange(gsize_and_lsize.first), cl::NDRange(gsize_and_lsize.second));
        }
        case 2: {
            const auto gsize_and_lsize_x = jitk::work_ranges(work_group_size_2dx, b[0]->size);
            const auto gsize_and_lsize_y = jitk::work_ranges(work_group_size_2dy, b[1]->size);
            return make_pair(cl::NDRange(gsize_and_lsize_x.first, gsize_and_lsize_y.first),
                             cl::NDRange(gsize_and_lsize_x.second, gsize_and_lsize_y.second));
        }
        case 3: {
            const auto gsize_and_lsize_x = jitk::work_ranges(work_group_size_3dx, b[0]->size);
            const auto gsize_and_lsize_y = jitk::work_ranges(work_group_size_3dy, b[1]->size);
            const auto gsize_and_lsize_z = jitk::work_ranges(work_group_size_3dz, b[2]->size);
            return make_pair(cl::NDRange(gsize_and_lsize_x.first, gsize_and_lsize_y.first, gsize_and_lsize_z.first),
                             cl::NDRange(gsize_and_lsize_x.second, gsize_and_lsize_y.second, gsize_and_lsize_z.second));
        }
        default:
            throw runtime_error("NDRanges: maximum of three dimensions!");
    }
}


cl::Program EngineOpenCL::getFunction(const string &source) {
    size_t hash = hasher(source);
    ++stat.kernel_cache_lookups;

    // Do we have the program already?
    if (_programs.find(hash) != _programs.end()) {
        return _programs.at(hash);
    }

    fs::path binfile = cache_bin_dir / jitk::hash_filename(compilation_hash, hash, ".clbin");
    cl::Program program;

    // If the binary file of the kernel doesn't exist we compile the source
    if (verbose or cache_bin_dir.empty() or not fs::exists(binfile)) {
        ++stat.kernel_cache_misses;
        program = cl::Program(context, source);
        if (verbose) {
            const string log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
            if (not log.empty()) {
                cout << "************ Build Log ************" << endl
                     << log
                     << "^^^^^^^^^^^^^ Log END ^^^^^^^^^^^^^" << endl << endl;
            }
            cout << "************ SOURCE ************" << endl
                 << source
                 << "^^^^^^^^^^^^^ SOURCE^^^^^^^^^^^^" << endl << endl;
        }
    } else { // If the binary file exist we load the binary into the program

        // First we load the binary into an vector
        vector<char> bin;
        {
            ifstream f(binfile.string(), ifstream::in | ifstream::binary);
            if (!f.is_open() or f.eof() or f.fail()) {
                throw runtime_error("Failed loading binary cache file");
            }
            f.seekg(0, std::ios_base::end);
            const std::streampos file_size = f.tellg();
            bin.resize(file_size);
            f.seekg(0, std::ios_base::beg);
            f.read(&bin[0], file_size);
        }

        // And then we load the binary into a program
        const vector<cl::Device> dev_list = {device};
        const cl::Program::Binaries bin_list = {make_pair(&bin[0], bin.size())};
        program = cl::Program(context, dev_list, bin_list);
    }

    // Finally, we build, save, and return the program
    try {
        program.build({device}, compile_flg.c_str());
    } catch (cl::Error &e) {
        cerr << "Error building: " << endl << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << endl;
        throw;
    }
    _programs[hash] = program;
    return program;
}

void EngineOpenCL::execute(const std::string &source, const std::vector<bh_base*> &non_temps,
                           const vector<const jitk::LoopB*> &threaded_blocks,
                           const vector<const bh_view*> &offset_strides,
                           const vector<const bh_instruction*> &constants) {
    auto tcompile = chrono::steady_clock::now();
    cl::Program program = getFunction(source);
    stat.time_compile += chrono::steady_clock::now() - tcompile;

    // Let's execute the OpenCL kernel
    cl::Kernel opencl_kernel = cl::Kernel(program, "execute");

    cl_uint i = 0;
    for (bh_base *base: non_temps) { // NB: the iteration order matters!
        opencl_kernel.setArg(i++, *getBuffer(base));
    }

    for (const bh_view *view: offset_strides) {
        uint64_t t1 = (uint64_t) view->start;
        opencl_kernel.setArg(i++, t1);
        for (int j=0; j<view->ndim; ++j) {
            uint64_t t2 = (uint64_t) view->stride[j];
            opencl_kernel.setArg(i++, t2);
        }
    }

    for (const bh_instruction *instr: constants) {
        switch (instr->constant.type) {
            case bh_type::BOOL:
                opencl_kernel.setArg(i++, instr->constant.value.bool8);
                break;
            case bh_type::INT8:
                opencl_kernel.setArg(i++, instr->constant.value.int8);
                break;
            case bh_type::INT16:
                opencl_kernel.setArg(i++, instr->constant.value.int16);
                break;
            case bh_type::INT32:
                opencl_kernel.setArg(i++, instr->constant.value.int32);
                break;
            case bh_type::INT64:
                opencl_kernel.setArg(i++, instr->constant.value.int64);
                break;
            case bh_type::UINT8:
                opencl_kernel.setArg(i++, instr->constant.value.uint8);
                break;
            case bh_type::UINT16:
                opencl_kernel.setArg(i++, instr->constant.value.uint16);
                break;
            case bh_type::UINT32:
                opencl_kernel.setArg(i++, instr->constant.value.uint32);
                break;
            case bh_type::UINT64:
                opencl_kernel.setArg(i++, instr->constant.value.uint64);
                break;
            case bh_type::FLOAT32:
                opencl_kernel.setArg(i++, instr->constant.value.float32);
                break;
            case bh_type::FLOAT64:
                opencl_kernel.setArg(i++, instr->constant.value.float64);
                break;
            case bh_type::COMPLEX64:
                opencl_kernel.setArg(i++, instr->constant.value.complex64);
                break;
            case bh_type::COMPLEX128:
                opencl_kernel.setArg(i++, instr->constant.value.complex128);
                break;
            case bh_type::R123:
                opencl_kernel.setArg(i++, instr->constant.value.r123);
                break;
            default:
                std::cerr << "Unknown OpenCL type: " << bh_type_text(instr->constant.type) << std::endl;
                throw std::runtime_error("Unknown OpenCL type");
        }
    }

    const auto ranges = NDRanges(threaded_blocks);
    auto texec = chrono::steady_clock::now();
    queue.enqueueNDRangeKernel(opencl_kernel, cl::NullRange, ranges.first, ranges.second);
    queue.finish();
    stat.time_exec += chrono::steady_clock::now() - texec;
}

void EngineOpenCL::set_constructor_flag(std::vector<bh_instruction*> &instr_list) {
    jitk::util_set_constructor_flag(instr_list, buffers);
}

std::string EngineOpenCL::info() const {
    stringstream ss;
    ss << "----"                                                                        << "\n";
    ss << "OpenCL:"                                                                     << "\n";
    ss << "  Platform: \"" << platform.getInfo<CL_PLATFORM_NAME>()                      << "\"\n";
    ss << "  Device:   \"" << device.getInfo<CL_DEVICE_NAME>() << " (" \
                           << device.getInfo<CL_DEVICE_OPENCL_C_VERSION>()              << ")\"\n";
    ss << "  Memory:   \"" << device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() / 1024 / 1024 << " MB\"\n";
    return ss.str();
}

} // bohrium
