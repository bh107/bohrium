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
namespace composite {

int64_t bh_get_integer(bh_constant constant)
{
    // TODO: Add extraction for additonal types.
    switch(constant.type) {
        case BH_FLOAT32:
            return (constant.value.float32) == (uint64_t)constant.value.float32 ? (int64_t)constant.value.float32 : -1;
            break;
        case BH_FLOAT64:
            return (constant.value.float64) == (uint64_t)constant.value.float64 ? (int64_t)constant.value.float64 : -1;
        default:
            return -1;
    }
}

int Expander::expand_powk(bh_ir& bhir, int pc)
{
    int start_pc = pc;                              
    bh_instruction& instr = bhir.instr_list[pc];    // Grab the BH_POWER instruction
    int64_t const k = 100;                          // Max exponent "unfolding"

    int64_t exponent = bh_get_integer(instr.constant);

    if (!bh_is_constant(&instr.operand[2])) {       // Transformation does not apply
        return 0;
    }

    if ((exponent < 1) || (exponent > k)) {         // Transformation does not apply
        return 0;
    }

    // TODO: Add support for this case by using intermediates.
    if (instr.operand[0].base == instr.operand[1].base) {
        return 0;
    }

    bh_view out = instr.operand[0];                 // Grab operands
    bh_view in1 = instr.operand[1];

    instr.opcode = BH_NONE;                         // Lazy choice... no re-use just NOP it.

    if (exponent == 0) {                                // x^0 = [1,1,...,1]
        inject(bhir, ++pc, BH_IDENTITY, out, 1);
    } else if (exponent == 1) {                         // x^1 = x
        inject(bhir, ++pc, BH_IDENTITY, out, in1);
    } else if (exponent == 2) {                         // x^2 = x*x
        inject(bhir, ++pc, BH_MULTIPLY, out, in1, in1);
    } else if (exponent == 3) {                         // x^3 = x*x*x
        inject(bhir, ++pc, BH_MULTIPLY, out, in1, in1); 
        inject(bhir, ++pc, BH_MULTIPLY, out, out, in1);
    } else if (exponent == 4) {                         // x^4 = (x*x)*(x*x)
        inject(bhir, ++pc, BH_MULTIPLY, out, in1, in1);
        inject(bhir, ++pc, BH_MULTIPLY, out, out, out);
    } else if (exponent == 5) {                         // x^5 = (x*x)*(x*x)*x
        inject(bhir, ++pc, BH_MULTIPLY, out, in1, in1);
        inject(bhir, ++pc, BH_MULTIPLY, out, out, out);
        inject(bhir, ++pc, BH_MULTIPLY, out, out, in1);
    } else {
        // TODO: Replace this with squaring.
        // Linear unroll.
        inject(bhir, ++pc, BH_MULTIPLY, out, in1, in1); // First multiplication
        for(int exp=2; exp<exponent; ++exp) {           // The remaining
            inject(bhir, ++pc, BH_MULTIPLY, out, out, in1);
        }

    }

    return pc-start_pc;
}

}}}
