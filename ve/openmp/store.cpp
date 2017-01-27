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

#include "store.hpp"

using namespace std;
namespace fs = boost::filesystem;

namespace bohrium {

static boost::hash<string> hasher;

Store::Store(const ConfigParser &config, jitk::Statistics &stat) :
                                           tmp_dir(fs::temp_directory_path() / fs::unique_path("bohrium_%%%%")),
                                           source_dir(tmp_dir / "src"),
                                           object_dir(tmp_dir / "obj"),
                                           compiler(config.defaultGet<string>("compiler_cmd", "/usr/bin/cc"),
                                                    config.defaultGet<string>("compiler_inc", ""),
                                                    config.defaultGet<string>("compiler_lib", "-lm"),
                                                    config.defaultGet<string>("compiler_flg", ""),
                                                    config.defaultGet<string>("compiler_ext", "")),
                                           verbose(config.defaultGet<bool>("verbose", false)),
                                           stat(stat)
{
    // Let's make sure that the directories exist
    fs::create_directories(source_dir);
    fs::create_directories(object_dir);
}

Store::~Store() {
    for(void *handle: _lib_handles) {
        dlerror(); // Reset errors
        if (dlclose(handle)) {
            cerr << dlerror() << endl;
        }
    }
}

// Returns the filename of the given 'hash'
static string hash_filename(size_t hash, string extension=".so") {
    stringstream ss;
    ss << setfill ('0') << setw(sizeof(size_t)*2) << hex << hash << extension;
    return ss.str();
}

KernelFunction Store::getFunction(const string &source) {
    size_t hash = hasher(source);
    ++stat.kernel_cache_lookups;

    // Do we have the function compiled and ready already?
    if (_functions.find(hash) != _functions.end()) {
        return _functions.at(hash);
    }
    ++stat.kernel_cache_misses;

    // The object file path
    fs::path objfile = object_dir / hash_filename(hash);

    // Write the source file and compile it (reading from disk)
    // NB: this is a nice debug option, but will hurt performance
    if (verbose) {
        fs::path srcfile = source_dir;
        {
            srcfile /= hash_filename(hash, ".c");
            ofstream ofs(srcfile.string());
            ofs << source;
            ofs.flush();
            ofs.close();
        }
        cout << "Write source " << srcfile << endl;
        compiler.compile(objfile.string(), srcfile.string());
    } else {
        // Pipe the source directly into the compiler thus no source file is written
        compiler.compile(objfile.string(), source.c_str(), source.size());
    }

    // Load the shared library
    void *lib_handle = dlopen(objfile.string().c_str(), RTLD_NOW);
    if (lib_handle == NULL) {
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
    if (dlsym_error) {
        cerr << "Cannot load function launcher(): " << dlsym_error << endl;
        throw runtime_error("VE-OPENMP: Cannot load function launcher()");
    }

    return _functions.at(hash);
}

} // bohrium

