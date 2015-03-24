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
#include <bh.h>
#include <stdio.h>

using namespace std;

typedef vector<bh_instruction> ilist;
typedef ilist::iterator ilist_iter;

void filter(bh_ir &bhir)
{
    for(ilist_iter it = bhir.instr_list.begin();
        it!=bhir.instr_list.end();
        ++it) {
        bh_instruction& instr = *it;
        switch(instr.opcode) {
            case BH_ADD_REDUCE:
            case BH_MULTIPLY_REDUCE:
            case BH_MINIMUM_REDUCE:
            case BH_MAXIMUM_REDUCE:
            case BH_LOGICAL_AND_REDUCE:
            case BH_BITWISE_AND_REDUCE:
            case BH_LOGICAL_OR_REDUCE:
            case BH_BITWISE_OR_REDUCE:
            case BH_LOGICAL_XOR_REDUCE:
            case BH_BITWISE_XOR_REDUCE:
                printf("Reduction...");
                break;

            case BH_FREE:
                printf("Free...");
                break;

            case BH_DISCARD:
                printf("Discard...");
                break;

            default:
                printf("Something else.");
                break;
        }
    }
}
