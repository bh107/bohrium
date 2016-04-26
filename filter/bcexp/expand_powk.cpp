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

int Expander::expand_powk(bh_ir& bhir, int pc)
{
    int start_pc = pc;
    bh_instruction& instr = bhir.instr_list[pc];        // Grab the BH_POWER instruction
    int64_t const k = 100;                              // Max exponent "unfolding"

    if (!bh_is_constant(&instr.operand[2])) {           // Transformation does not apply
        return 0;
    }

    if (!bh_type_is_integer(instr.constant.type)) {
        return 0;
    }

    int64_t exponent;
    try {
        exponent = instr.constant.get_int64();          // Extract the exponent
    } catch (overflow_error& e) {
        return 0; //Give up, if we cannot get a signed integer
    }

    if (0 > exponent || exponent > k) {
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
