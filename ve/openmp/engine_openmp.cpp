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
#include <fstream>
#include <string>
#include <map>
#include <boost/functional/hash.hpp>
#include <iomanip>
#include <dlfcn.h>
#include <jitk/codegen_util.hpp>
#include <thread>

#include <bh_util.hpp>
#include "engine_openmp.hpp"

using namespace std;
namespace fs = boost::filesystem;

namespace bohrium {

static boost::hash<string> hasher;

EngineOpenMP::EngineOpenMP(const ConfigParser &config, jitk::Statistics &stat) :
                                           verbose(config.defaultGet<bool>("verbose", false)),
                                           cache_file_max(config.defaultGet<int64_t>("cache_file_max", 50000)),
                                           tmp_dir(jitk::get_tmp_path(config)),
                                           tmp_src_dir(tmp_dir / "src"),
                                           tmp_bin_dir(tmp_dir / "obj"),
                                           cache_bin_dir(fs::path(config.defaultGet<string>("cache_dir", ""))),
                                           compiler(config.get<string>("compiler_cmd"), verbose),
                                           compilation_hash(hasher(compiler.cmd_template)),
                                           stat(stat)
{
    // Let's make sure that the directories exist
    jitk::create_directories(tmp_src_dir);
    jitk::create_directories(tmp_bin_dir);
    if (not cache_bin_dir.empty()) {
        jitk::create_directories(cache_bin_dir);
    }
}

EngineOpenMP::~EngineOpenMP() {

    // Move JIT kernels to the cache dir
    if (not cache_bin_dir.empty()) {
     //   cout << "filling cache_bin_dir: " << cache_bin_dir.string() << endl;
        for (const auto &kernel: _functions) {
            const fs::path src = tmp_bin_dir / jitk::hash_filename(compilation_hash, kernel.first, ".so");
            if (fs::exists(src)) {
                const fs::path dst = cache_bin_dir / jitk::hash_filename(compilation_hash, kernel.first, ".so");
                if (not fs::exists(dst)) {
                    fs::copy(src, dst);
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

    // If this cleanup is enabled, the application segfaults
    // on destruction of the EngineOpenMP class.
    //
    // The reason for that is that OpenMP currently has no
    // means in the definition of the standard to do a proper
    // cleanup and hence internally does something funny.
    // See https://stackoverflow.com/questions/5200418/destroying-threads-in-openmp-c
    // for details.
    //
    // TODO A more sensible way to get around this issue
    //      would be to also dlopen the openmp library
    //      itself and therefore increase our reference count
    //      to it. This enables the kernel so files to be unlinked
    //      and cleaned up, but prevents the problematic code on
    //      openmp cleanup to be triggered at bohrium runtime.

    // for(void *handle: _lib_handles) {
    //     dlerror(); // Reset errors
    //     if (dlclose(handle)) {
    //         cerr << dlerror() << endl;
    //     }
    // }
}

KernelFunction EngineOpenMP::getFunction(const string &source) {
    size_t hash = hasher(source);
    ++stat.kernel_cache_lookups;

    // Do we have the function compiled and ready already?
    if (_functions.find(hash) != _functions.end()) {
        return _functions.at(hash);
    }

    fs::path binfile = cache_bin_dir / jitk::hash_filename(compilation_hash, hash, ".so");

    // If the binary file of the kernel doesn't exist we create it
    if (verbose or cache_bin_dir.empty() or not fs::exists(binfile)) {
        ++stat.kernel_cache_misses;

        // We create the binary file in the tmp dir
        binfile = tmp_bin_dir / jitk::hash_filename(compilation_hash, hash, ".so");

        // Write the source file and compile it (reading from disk)
        // NB: this is a nice debug option, but will hurt performance
        if (verbose) {
            fs::path srcfile = jitk::write_source2file(source, tmp_src_dir,
                                                       jitk::hash_filename(compilation_hash, hash, ".c"),
                                                       true);
            compiler.compile(binfile.string(), srcfile.string());
        } else {
            // Pipe the source directly into the compiler thus no source file is written
            compiler.compile(binfile.string(), source.c_str(), source.size());
        }
    }

    // Load the shared library
    void *lib_handle = dlopen(binfile.string().c_str(), RTLD_NOW);
    if (lib_handle == nullptr) {
        cerr << "Cannot load library: " << dlerror() << endl;
        throw runtime_error("VE-OPENMP: Cannot load library");
    }
    _lib_handles.push_back(lib_handle);

    // Load the launcher function
    // The (clumsy) cast conforms with the ISO C standard and will
    // avoid any compiler warnings.
    dlerror(); // Reset errors
    *(void **) (&_functions[hash]) = dlsym(lib_handle, "launcher");
    const char* dlsym_error = dlerror();
    if (dlsym_error != nullptr) {
        cerr << "Cannot load function launcher(): " << dlsym_error << endl;
        throw runtime_error("VE-OPENMP: Cannot load function launcher()");
    }
    return _functions.at(hash);
}


void EngineOpenMP::execute(const std::string &source, const std::vector<bh_base*> &non_temps,
                           const std::vector<const bh_view*> &offset_strides,
                           const std::vector<const bh_instruction*> &constants) {

    // Make sure all arrays are allocated
    for (bh_base *base: non_temps) {
        bh_data_malloc(base);
    }

    // Compile the kernel
    auto tbuild = chrono::steady_clock::now();
    KernelFunction func = getFunction(source);
    assert(func != NULL);
    stat.time_compile += chrono::steady_clock::now() - tbuild;

    // Create a 'data_list' of data pointers
    vector<void*> data_list;
    data_list.reserve(non_temps.size());
    for(bh_base *base: non_temps) {
        assert(base->data != NULL);
        data_list.push_back(base->data);
    }

    // And the offset-and-strides
    vector<uint64_t> offset_and_strides;
    offset_and_strides.reserve(offset_strides.size());
    for (const bh_view *view: offset_strides) {
        const uint64_t t = (uint64_t) view->start;
        offset_and_strides.push_back(t);
        for (int i=0; i<view->ndim; ++i) {
            const uint64_t s = (uint64_t) view->stride[i];
            offset_and_strides.push_back(s);
        }
    }

    // And the constants
    vector<bh_constant_value> constant_arg;
    constant_arg.reserve(constants.size());
    for (const bh_instruction* instr: constants) {
        constant_arg.push_back(instr->constant.value);
    }

    auto texec = chrono::steady_clock::now();
    // Call the launcher function, which will execute the kernel
    func(&data_list[0], &offset_and_strides[0], &constant_arg[0]);
    stat.time_exec += chrono::steady_clock::now() - texec;

}

void EngineOpenMP::set_constructor_flag(std::vector<bh_instruction*> &instr_list) {
    const std::set<bh_base*> empty;
    jitk::util_set_constructor_flag(instr_list, empty);
}


std::string EngineOpenMP::info() const {
    stringstream ss;
    ss << "----"                                                           << "\n";
    ss << "OpenMP:"                                                        << "\n";
    ss << "  Hardware threads: " << std::thread::hardware_concurrency()    << "\n";
    ss << "  JIT Command: \"" << compiler.cmd_template << "\"\n";
    return ss.str();
}

} // bohrium
