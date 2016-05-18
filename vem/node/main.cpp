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
#include <bh_component.hpp>

using namespace bohrium;
using namespace component;
using namespace std;

namespace {
class Impl : public ComponentImplWithChild {
  private:
    //Allocated base arrays
    set<bh_base*> _allocated_bases;
    //Inspect one instruction and throws exception on error
    void inspect(bh_instruction *instr);
  public:
    Impl(int stack_level) : ComponentImplWithChild(stack_level) {}
    ~Impl(); // NB: a destructor implementation must exist
    void execute(bh_ir *bhir);
};
} //Unnamed namespace

extern "C" ComponentImpl* create(int stack_level) {
    return new Impl(stack_level);
}
extern "C" void destroy(ComponentImpl* self) {
    delete self;
}

Impl::~Impl() {
    if (_allocated_bases.size() > 0)
    {
        long s = (long) _allocated_bases.size();
        if (s > 20) {
            cerr << "[NODE-VEM] Warning " << s << " base arrays were not discarded "
                    "on exit (too many to show here)." << endl;
        } else {
            cerr << "[NODE-VEM] Warning " << s << " base arrays were not discarded "
                    "on exit (only showing the array IDs because the view list may "
                    "be corrupted due to reuse of base struct):" << endl;
            for (bh_base *b: _allocated_bases) {
                cerr << *b << endl;
            }
        }
    }
}

void Impl::inspect(bh_instruction *instr) {
    int nop = bh_noperands(instr->opcode);
    bh_view *operands = bh_inst_operands(instr);

    //Save all new base arrays
    for(bh_intp o=0; o<nop; ++o)
    {
        if(!bh_is_constant(&operands[o]))
            _allocated_bases.insert(operands[o].base);
    }

    //And remove discarded arrays
    if(instr->opcode == BH_DISCARD)
    {
        bh_base *base = operands[0].base;
        if(_allocated_bases.erase(base) != 1)
        {
            cerr << "[NODE-VEM] discarding unknown base array: " << *base << endl;
            throw runtime_error("[NODE-VEM] discarding unknown base array");
        }
    }
}

void Impl::execute(bh_ir *bhir) {
    for(uint64_t i=0; i < bhir->instr_list.size(); ++i)
        inspect(&bhir->instr_list[i]);
    child.execute(bhir);
}

