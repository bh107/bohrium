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
#include <memory> // For unique_ptr

#include <bh_component.hpp>
#include <bh_extmethod.hpp>

using namespace std;
using namespace bohrium;
using namespace component;
using namespace extmethod;

namespace {
class Impl : public ComponentImplWithChild {
  private:
    //The visualizer opcode and implementation
    bh_opcode viz_opcode = -1;
    unique_ptr<ExtmethodFace> viz_impl;

    //Checks if the visualizer opcode exists in the bhir
    bool visualizer_in_bhir(const bh_ir *bhir) {
        for(const bh_instruction &instr: bhir->instr_list) {
            if(instr.opcode == viz_opcode)
                return true;
        }
        return false;
    }
  public:
    Impl(int stack_level) : ComponentImplWithChild(stack_level) {}
    ~Impl() {}; // NB: a destructor implementation must exist
    void execute(bh_ir *bhir);
    void extmethod(const string &name, bh_opcode opcode) {
        viz_impl.reset(new ExtmethodFace(config, name));
        viz_opcode = opcode;
    }
};
} //Unnamed namespace

extern "C" ComponentImpl* create(int stack_level) {
    return new Impl(stack_level);
}
extern "C" void destroy(ComponentImpl* self) {
    delete self;
}

void Impl::execute(bh_ir *bhir) {
    if(not visualizer_in_bhir(bhir))
        return child.execute(bhir);

    size_t exec_count = 0;//Count of already executed instruction
    for(size_t i=0; i<bhir->instr_list.size(); ++i)
    {
        const bh_instruction &instr = bhir->instr_list[i];
        if(instr.opcode == viz_opcode)
        {
            bh_instruction sync[2];
            sync[0].opcode = BH_SYNC;
            sync[0].operand[0] = instr.operand[0];
            sync[1].opcode = BH_SYNC;
            sync[1].operand[0] = instr.operand[1];

            if(exec_count < i)//Let's execute the instructions between 'exec_count' and 'i' with an appended SYNC
            {
                bh_ir b(i - exec_count, &bhir->instr_list[exec_count]);
                b.instr_list.push_back(sync[0]);
                b.instr_list.push_back(sync[1]);
                child.execute(&b);
            }
            else
            {
                bh_ir b(2, sync);
                child.execute(&b);
            }
            //Now let's visualize
            assert(viz_impl);
            viz_impl->execute(&bhir->instr_list[i], NULL);
            exec_count = i;
        }
    }
    if(bhir->instr_list.size() > exec_count)
    {
        bh_ir b(bhir->instr_list.size() - exec_count, &bhir->instr_list[exec_count]);
        child.execute(&b);
    }
}

