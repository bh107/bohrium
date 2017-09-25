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

#include <bh_component.hpp>

using namespace bohrium;
using namespace component;
using namespace std;

namespace {
int remove_none(vector<bh_instruction> &bh_instr_list, int from, int to, int decrementer)
{
    int i = from;

    // Loop through the partial instruction list. Increment happens inside.
    for(auto it = bh_instr_list.begin() + from; it != bh_instr_list.begin() + to; ++i)
    {
        switch(it->opcode) {
            case BH_REPEAT:
                // Recursively remove BH_NONE from inner part of BH_REPEAT
                decrementer = remove_none(bh_instr_list, i + 1, i + 1 + it->constant.value.r123.start, decrementer);
                it->constant.value.r123.start -= decrementer;

                // Increment iterator manually
                ++it;
                break;
            case BH_NONE:
                // Remove the BH_NONE and increase the decrementer for inner BH_REPEAT
                it = bh_instr_list.erase(it);
                ++decrementer;
                break;
            default:
                // Increment iterator manually
                ++it;
        }
    }

    return decrementer;
}

class Impl : public ComponentImplWithChild {
  public:
    Impl(int stack_level) : ComponentImplWithChild(stack_level) {};
    ~Impl() {}; // NB: a destructor implementation must exist
    void execute(BhIR *bhir) {
        // Remove BH_NONE from entire instruction list
        remove_none(bhir->instr_list, 0, bhir->instr_list.size(), 0);
        child.execute(bhir);
    };
};
} //Unnamed namespace

extern "C" ComponentImpl* create(int stack_level) {
    return new Impl(stack_level);
}
extern "C" void destroy(ComponentImpl* self) {
    delete self;
}
