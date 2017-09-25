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
namespace bcexp {

static const int64_t max_exponent_unfolding = 100;

int Expander::expand_powk(BhIR& bhir, int pc)
{
    verbose_print("[Powk] Expanding BH_POWER");

    int start_pc = pc;

    // Grab the BH_POWER instruction
    bh_instruction& instr = bhir.instr_list[pc];

    // Transformation does not apply for non constants
    if (!bh_is_constant(&instr.operand[2])) {
        return 0;
    }

    if (!bh_type_is_integer(instr.constant.type)) {
        return 0;
    }

    int64_t exponent;
    try {
        // Extract the exponent
        exponent = instr.constant.get_int64();
    } catch (overflow_error& e) {
        // Give up, if we cannot get a signed integer
        verbose_print("[Powk] \tCan't expand BH_POWER with non-integer");
        return 0;
    }

    if (0 > exponent || exponent > max_exponent_unfolding) {
        verbose_print("[Powk] \tCan't expand BH_POWER with exponent " + std::to_string(exponent));
        return 0;
    }

    // TODO: Add support for this case by using intermediates.
    if (instr.operand[0].base == instr.operand[1].base) {
        verbose_print("[Powk] \tCan't expand BH_POWER without intermediates.");
        return 0;
    }

    // Lazy choice... no re-use just NOP it.
    instr.opcode = BH_NONE;

    // Grab operands
    bh_view out = instr.operand[0];
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

    return pc - start_pc;
}

}}}
