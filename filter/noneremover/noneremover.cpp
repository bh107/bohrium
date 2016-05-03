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

#include <bh_component.h>

using namespace std;

void noneremover_filter(bh_ir &bhir)
{
    int decrementer = 0;

    // Loop through the instruction list. Increment happens inside.
    for(auto it = bhir.instr_list.begin(); it != bhir.instr_list.end(); /* !!! */)
    {
        switch(it->opcode) {
            case BH_REPEAT:
                // We need to loop through the inner parts of the repeat
                // TODO: Nested BH_REPEATs
                decrementer = 0;

                for(int i = 1; i <= it->constant.value.r123.start; ++i) {
                    if ((it+i)->opcode == BH_NONE) {
                        it = bhir.instr_list.erase(it+i);
                        ++decrementer;
                    }
                }

                // Decrement BH_REPEAT's size
                it->constant.value.r123.start -= decrementer;

                // Move iterator to end of BH_REPEAT
                it += it->constant.value.r123.start;

                break;
            case BH_NONE:
                // Remove the BH_NONE
                it = bhir.instr_list.erase(it);
                break;
            default:
                // Increment iterator manually
                ++it;
        }
    }
}
