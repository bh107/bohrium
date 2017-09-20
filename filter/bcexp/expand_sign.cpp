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

/**
 *  Expand BH_SIGN at the given PC into the sequence:
 *
 *          BH_SIGN OUT, IN1 (When IN1.type != COMPLEX):
 *
 *  LESS, t1_bool, input, 0.0
 *  IDENTITY, t1, t1_bool
 *  FREE, t1_bool
 *
 *  GREATER, t2_bool, input, 0.0
 *  IDENTITY, t2, t2_bool
 *  FREE, t2_bool
 *
 *  SUBTRACT, out, t2, t1
 *  FREE, t1
 *  FREE, t2
 *
 *          BH_SIGN OUT, IN1 (When IN1.type == COMPLEX):
 *
 *  REAL, input_r, input
 *
 *  LESS, t1_bool, input_r, 0.0
 *  IDENTITY, t1, t1_bool
 *  FREE, t1_bool
 *
 *  GREATER, t2_bool, input_r, 0.0
 *  FREE, input_r
 *
 *  IDENTITY, t2, t2_bool
 *  FREE, t2_bool
 *
 *  SUBTRACT, t3, t2, t1
 *  FREE, t1
 *  FREE, t2
 *
 *  IDENTITY, out, t3
 *  FREE, t3
 *
 *  Returns the number of instructions used (12 or 17).
 */
int Expander::expand_sign(BhIR& bhir, int pc)
{
    int start_pc = pc;
    bh_instruction& composite = bhir.instr_list[pc];

    // Lazy choice... no re-use just NOP it.
    composite.opcode = BH_NONE;

    // Grab operands
    bh_view output = composite.operand[0];
    bh_view input  = composite.operand[1];

    // Grab the input-type
    bh_type input_type = input.base->type;

    // Inherit ndim and shape
    bh_view meta = composite.operand[0];
    meta.start = 0;

    // Count number of elements
    int64_t nelements = 1;

    // Contiguous stride
    for(int64_t dim=meta.ndim-1; dim >= 0; --dim) {
        meta.stride[dim] = nelements;
        nelements *= meta.shape[dim];
    }

    if (!((input_type == bh_type::COMPLEX64) || (input_type == bh_type::COMPLEX128))) {
        verbose_print("[Sign] Expanding complex sign");
        // For non-complex: sign(x) = (x>0)-(x<0)
        // Temps
        bh_view lss    = make_temp(meta, input_type, nelements);
        bh_view gtr    = make_temp(meta, input_type, nelements);
        bh_view t_bool = make_temp(meta, bh_type::BOOL,    nelements);

        // Sequence
        inject(bhir, ++pc, BH_GREATER,  t_bool, input, 0.0);
        inject(bhir, ++pc, BH_IDENTITY, lss,    t_bool);
        inject(bhir, ++pc, BH_FREE,     t_bool);

        inject(bhir, ++pc, BH_LESS,     t_bool, input, 0.0);
        inject(bhir, ++pc, BH_IDENTITY, gtr,    t_bool);
        inject(bhir, ++pc, BH_FREE,     t_bool);

        inject(bhir, ++pc, BH_SUBTRACT, output, lss, gtr);
        inject(bhir, ++pc, BH_FREE,     lss);
        inject(bhir, ++pc, BH_FREE,     gtr);
    } else {
        verbose_print("[Sign] Expanding normal sign");
        // For complex: sign(0) = 0, sign(z) = z/|z|
        bh_type float_type = (input_type == bh_type::COMPLEX64) ? bh_type::FLOAT32 : bh_type::FLOAT64;

        // General form: sign(z) = z/(|z|+(z==0))
        // Temps
        bh_view f_abs  = make_temp(meta, float_type, nelements);
        bh_view b_zero = make_temp(meta, bh_type::BOOL,    nelements);
        bh_view f_zero = make_temp(meta, float_type, nelements);

        // Sequence
        inject(bhir, ++pc, BH_ABSOLUTE, f_abs,  input);
        inject(bhir, ++pc, BH_EQUAL,    b_zero, f_abs, 0.0, float_type);
        inject(bhir, ++pc, BH_IDENTITY, f_zero, b_zero);
        inject(bhir, ++pc, BH_FREE,     b_zero);

        inject(bhir, ++pc, BH_ADD,     f_abs, f_abs, f_zero);
        inject(bhir, ++pc, BH_FREE,    f_zero);

        inject(bhir, ++pc, BH_IDENTITY, output, f_abs);
        inject(bhir, ++pc, BH_FREE,     f_abs);

        inject(bhir, ++pc, BH_DIVIDE, output, input, output);
    }

    return pc - start_pc;
}

}}}
