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

#include <dlfcn.h>
#include <string>
#include <vector>
#include <sstream>

#include <bh_extmethod.hpp>

using namespace std;

namespace bohrium {
namespace extmethod {

ExtmethodFace::ExtmethodFace(const ConfigParser &parent_config,
                             const string &name) : _name(name) {

    vector<boost::filesystem::path> libs = parent_config.getListOfPaths("libs");
    const string name_func_create = name + "_create";
    const string name_func_destroy = name + "_destroy";
    stringstream err_msg;

    bool fail = true;
    for (const boost::filesystem::path &lib_path: libs) {
        // Load the shared library
        _lib_handle = dlopen(lib_path.string().c_str(), RTLD_NOW);
        if (_lib_handle == NULL) {
            cerr << "Cannot load library: " << dlerror() << '\n';
            throw runtime_error("Extmethod: Cannot load library");
        }

        // Load the extmethod's create and destroy functions
        // The (clumsy) cast conforms with the ISO C standard and will
        // avoid any compiler warnings.
        bool found = true;
        {
            dlerror(); // Reset errors
            *(void **) (&_create) = dlsym(_lib_handle, name_func_create.c_str());
            const char* dlsym_error = dlerror();
            if (dlsym_error) {
                err_msg << "Failed loading '" << name_func_create << "' in "
                        << lib_path << ": " << dlsym_error << endl;
                found = false;
            }
        }
        {
            dlerror(); // Reset errors
            *(void **) (&_destroy) = dlsym(_lib_handle, name_func_destroy.c_str());
            const char* dlsym_error = dlerror();
            if (dlsym_error) {
                err_msg << "Failed loading '" << name_func_destroy << "' in "
                        << lib_path << ": " << dlsym_error << endl;
                found = false;
            }
        }
        if (found) {
            fail = false;
            break;
        } else {
            dlerror(); // Reset errors
            if (dlclose(_lib_handle)) {
                cerr << dlerror() << endl;
                throw runtime_error("Extmethod: Cannot close library");
            }
        }
    }
    if (fail) {
        err_msg << "Extmethod: Cannot find '" << _name << "':" << endl;
        throw ExtmethodNotFound(err_msg.str());
    }
    _implementation = _create();
}

ExtmethodFace::~ExtmethodFace() {
    if (_implementation != nullptr) {
        _destroy(_implementation);
        dlerror(); // Reset errors
        assert(_lib_handle != nullptr);
        if (dlclose(_lib_handle)) {
            cerr << dlerror() << endl;
        }
    }
}

}} //namespace bohrium::extmethod
