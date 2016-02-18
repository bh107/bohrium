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
#include <math.h>

using namespace std;

namespace bohrium {
namespace filter {
namespace composite {

int64_t bh_get_integer(bh_constant constant)
{
    switch(constant.type) {
        case BH_UINT8:
            return (int64_t)constant.value.uint8;
        case BH_UINT16:
            return (int64_t)constant.value.uint16;
        case BH_UINT32:
            return (int64_t)constant.value.uint32;
        case BH_UINT64:
            return (int64_t)constant.value.uint64;

        case BH_INT8:
            return (constant.value.int8) == (uint64_t)constant.value.int8 ? (int64_t)constant.value.int8 : -1;
        case BH_INT16:
            return (constant.value.int16) == (uint64_t)constant.value.int16 ? (int64_t)constant.value.int16 : -1;
        case BH_INT32:
            return (constant.value.int32) == (uint64_t)constant.value.int32 ? (int64_t)constant.value.int32 : -1;
        case BH_INT64:
            return (constant.value.int64) == (uint64_t)constant.value.int64 ? (int64_t)constant.value.int64 : -1;

        case BH_FLOAT32:
            return (constant.value.float32) == (uint64_t)constant.value.float32 ? (int64_t)constant.value.float32 : -1;
        case BH_FLOAT64:
            return (constant.value.float64) == (uint64_t)constant.value.float64 ? (int64_t)constant.value.float64 : -1;
            
        default:
            return -1;
    }
}

int Expander::expand_powk(bh_ir& bhir, int pc)
{
    int start_pc = pc;
    bh_instruction& instr = bhir.instr_list[pc];        // Grab the BH_POWER instruction
    int64_t const k = 100;                              // Max exponent "unfolding"

    if (!bh_is_constant(&instr.operand[2])) {           // Transformation does not apply
        return 0;
    }

    int64_t exponent = bh_get_integer(instr.constant);  // Extract the exponent
    if ((exponent < 0) || (exponent > k)) {             // Transformation does not apply
        return 0;
    }

    // TODO: Add support for this case by using intermediates.
    if (instr.operand[0].base == instr.operand[1].base) {// Transformation does not apply
        return 0;
    }

    instr.opcode = BH_NONE;             // Lazy choice... no re-use just NOP it.
    bh_view out = instr.operand[0];     // Grab operands
    bh_view in1 = instr.operand[1];

    // Transform BH_POWER into BH_MULTIPLY sequences.
    if (exponent == 0) {                                // x^0 = [1,1,...,1]
        inject(bhir, ++pc, BH_IDENTITY, out, 1);
    } else if (exponent == 1) {                         // x^1 = x
        inject(bhir, ++pc, BH_IDENTITY, out, in1);
    } else {                                            // x^n = (x*x)*(x*x)*...
        int highest_power_below_input = pow(2, (int)log2(exponent));
        exponent -= highest_power_below_input;

        // Do x=x^2 as many times as n is a power of 2
        inject(bhir, ++pc, BH_MULTIPLY, out, in1, in1);
        highest_power_below_input /= 2;

        while(highest_power_below_input != 1) {
            inject(bhir, ++pc, BH_MULTIPLY, out, out, out);
            highest_power_below_input /= 2;
        }

        // Linear unroll the rest
        for(int exp=0; exp<exponent; ++exp) {
            inject(bhir, ++pc, BH_MULTIPLY, out, out, in1);
        }
    }

    return pc-start_pc;
}

}}}
