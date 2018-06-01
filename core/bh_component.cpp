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
    if (_lib_handle == nullptr) {
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
    if (initiated()) {
        _destroy(_implementation);
        dlerror(); // Reset errors
        if (dlclose(_lib_handle)) {
            cerr << dlerror() << endl;
        }
    }
}

bool ComponentFace::initiated() const {
    return _implementation != nullptr;
}

void ComponentFace::execute(BhIR *bhir) {
    if (not initiated()) {
        throw std::runtime_error("uninitiated component interface");
    }
    _implementation->execute(bhir);
}

void ComponentFace::extmethod(const std::string &name, bh_opcode opcode) {
    if (not initiated()) {
        throw std::runtime_error("uninitiated component interface");
    }
    _implementation->extmethod(name, opcode);
}

std::string ComponentFace::message(const std::string &msg) {
    if (not initiated()) {
        throw std::runtime_error("uninitiated component interface");
    }
    return _implementation->message(msg);
}

void *ComponentFace::getMemoryPointer(bh_base &base, bool copy2host, bool force_alloc, bool nullify) {
    if (not initiated()) {
        throw std::runtime_error("uninitiated component interface");
    }
    return _implementation->getMemoryPointer(base, copy2host, force_alloc, nullify);
}

void ComponentFace::setMemoryPointer(bh_base *base, bool host_ptr, void *mem) {
    if (not initiated()) {
        throw std::runtime_error("uninitiated component interface");
    }
    return _implementation->setMemoryPointer(base, host_ptr, mem);
}

void ComponentFace::memCopy(bh_view &src, bh_view &dst, const std::string &param) {
    if (not initiated()) {
        throw std::runtime_error("uninitiated component interface");
    }
    return _implementation->memCopy(src, dst, param);
}

void *ComponentFace::getDeviceContext() {
    if (not initiated()) {
        throw std::runtime_error("uninitiated component interface");
    }
    return _implementation->getDeviceContext();
}

void ComponentFace::setDeviceContext(void *device_context) {
    if (not initiated()) {
        throw std::runtime_error("uninitiated component interface");
    }
    _implementation->setDeviceContext(device_context);
}


}} //namespace bohrium::component
