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

#include <bh_component.hpp>

using namespace std;

namespace bohrium {
namespace component {

ComponentFace::ComponentFace(const string &lib_path, int stack_level) {

    // Load the shared library
    _lib_handle = dlopen(lib_path.c_str(), RTLD_NOW);
    if (_lib_handle == NULL) {
        cerr << "Cannot load library: " << dlerror() << '\n';
        throw runtime_error("ConfigParser: Cannot load library");
    }

    // Load the component's create and destroy functions
    // The (clumsy) cast conforms with the ISO C standard and will
    // avoid any compiler warnings.
    {
        dlerror(); // Reset errors
        *(void **) (&_create) = dlsym(_lib_handle, "create");
        const char* dlsym_error = dlerror();
        if (dlsym_error) {
            cerr << "Cannot load function 'create': " << dlsym_error << '\n';
            throw runtime_error("ComponentInterface: Cannot load function 'create'");
        }
    }
    {
        dlerror();
        *(void **) (&_destroy) = dlsym(_lib_handle, "destroy");
        const char* dlsym_error = dlerror();
        if (dlsym_error) {
            cerr << "Cannot load function 'destroy': " << dlsym_error << '\n';
            throw runtime_error("ComponentInterface: Cannot load function 'destroy'");
        }
    }
    _implementation = _create(stack_level);
}

ComponentFace::~ComponentFace() {
    _destroy(_implementation);
    dlerror(); // Reset errors
    if (dlclose(_lib_handle)) {
        cerr << dlerror() << endl;
    }
}


}} //namespace bohrium::component
