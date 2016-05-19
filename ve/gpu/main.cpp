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

#include <iostream>
#include <stdexcept>

#include <bh_component.hpp>
#include <bh_extmethod.hpp>

#include "InstructionScheduler.hpp"

using namespace std;
using namespace bohrium;
using namespace component;
using namespace extmethod;

ResourceManager* resourceManager;

namespace {
class Impl : public ComponentImplWithChild {
  private:
    InstructionScheduler instructionScheduler;
  public:
    Impl(int stack_level) : ComponentImplWithChild(stack_level) {
        resourceManager = new ResourceManager(&config, &child);
        // defined in core/bh_fuse.cpp
        std::string preferedFuseModel("NO_XSWEEP_SCALAR_SEPERATE_SHAPE_MATCH");
        char* fm = getenv("BH_FUSE_MODEL");
        if (fm != NULL) {
            std::string fuseModel(fm);
            if (preferedFuseModel.compare(fuseModel) != 0)
                std::cerr << "[GPU-VE] Warning! fuse model not set by the GPU-VE: '" <<
                fuseModel << std::endl;
        }
        else{
            setenv("BH_FUSE_MODEL", preferedFuseModel.c_str(), 1);
        }
    }
    ~Impl() {
        delete resourceManager;
    }
    void execute(bh_ir *bhir) {
        instructionScheduler.schedule(bhir);
    };
    void extmethod(const string &name, bh_opcode opcode) {
        // ExtmethodFace does not have a default or copy constructor thus
        // we have to use its move constructor.
        instructionScheduler.registerFunction(opcode, new ExtmethodFace(config, name));
    }
};
} //Unnamed namespace

extern "C" ComponentImpl* create(int stack_level) {
    return new Impl(stack_level);
}
extern "C" void destroy(ComponentImpl* self) {
    delete self;
}


