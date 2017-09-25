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
#include "expander.hpp"

using namespace std;

namespace bohrium {
namespace filter {
namespace bcexp {

int Expander::expand_repeat(BhIR& bhir, int pc)
{
    verbose_print("[Repeat] Expanding BH_REPEAT");
    // Grab the BH_REPEAT instruction
    bh_instruction& instr = bhir.instr_list[pc];

    // Get the two arguments, which are enclosed in the constant of type BH_R123
    int size = instr.constant.value.r123.start;
    int occur = instr.constant.value.r123.key;

    // Remove BH_REPEAT
    instr.opcode = BH_NONE;

    // Repeat content of BH_REPEAT occur-1 times.
    // -1 since it's already there once
    for (int i = 0; i < occur-1; ++i) {
        for (int j = 0; j < size; ++j) {
            ++pc;
            inject(bhir, pc, bhir.instr_list[pc + j]);
        }
    }

    return size * occur;
}

}}}
