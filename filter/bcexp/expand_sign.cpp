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

int Expander::expand_sign(bh_ir& bhir, int pc)
{
    int start_pc = pc;
    bh_instruction& composite = bhir.instr_list[pc];
    composite.opcode = BH_NONE; // Lazy choice... no re-use just NOP it.

    bh_view output  = composite.operand[0];         // Grab operands
    bh_view input   = composite.operand[1];

    bh_type input_type = input.base->type;          // Grab the input-type

    bh_view meta = composite.operand[0];            // Inherit ndim and shape
    meta.start = 0;
    bh_intp nelements = 1;                          // Count number of elements
    for(bh_intp dim=meta.ndim-1; dim >= 0; --dim) { // Contiguous stride
        meta.stride[dim] = nelements;
        nelements *= meta.shape[dim];
    }
    if (!((input_type == BH_COMPLEX64) || \
          (input_type == BH_COMPLEX128))) { // For non-complex: sign(x) = (x>0)-(x<0)
                                                            
        bh_view lss     = make_temp(meta, input_type, nelements);// Temps
        bh_view gtr     = make_temp(meta, input_type, nelements);
        bh_view t_bool  = make_temp(meta, BH_BOOL, nelements);  

        inject(bhir, ++pc, BH_GREATER, t_bool, input, 0.0);    // Sequence
        inject(bhir, ++pc, BH_IDENTITY, lss, t_bool);
        inject(bhir, ++pc, BH_FREE, t_bool);
        inject(bhir, ++pc, BH_DISCARD, t_bool);
        
        inject(bhir, ++pc, BH_LESS, t_bool, input, 0.0);       
        inject(bhir, ++pc, BH_IDENTITY, gtr, t_bool);
        inject(bhir, ++pc, BH_FREE, t_bool);
        inject(bhir, ++pc, BH_DISCARD, t_bool);

        inject(bhir, ++pc, BH_SUBTRACT, output, lss, gtr);
        inject(bhir, ++pc, BH_FREE, lss);
        inject(bhir, ++pc, BH_DISCARD, lss);
        inject(bhir, ++pc, BH_FREE, gtr);
        inject(bhir, ++pc, BH_DISCARD, gtr);
    } else {                                // For complex: sign(0) = 0, sign(z) = z/|z|

        bh_type float_type = (input_type == BH_COMPLEX64) ? BH_FLOAT32 : BH_FLOAT64;
                                            // General form: sign(z) = z/(|z|+(z==0))
        bh_view f_abs = make_temp(meta, float_type, nelements); // Temps
        bh_view b_zero = make_temp(meta, BH_BOOL, nelements);
        bh_view f_zero = make_temp(meta, float_type, nelements);

        inject(bhir, ++pc, BH_ABSOLUTE, f_abs, input);          // Sequence
        inject(bhir, ++pc, BH_EQUAL, b_zero, f_abs, 0.0, float_type);
        inject(bhir, ++pc, BH_IDENTITY, f_zero, b_zero);
        inject(bhir, ++pc, BH_FREE, b_zero);
        inject(bhir, ++pc, BH_DISCARD, b_zero);
        inject(bhir, ++pc, BH_ADD, f_abs, f_abs, f_zero);
        inject(bhir, ++pc, BH_FREE, f_zero);
        inject(bhir, ++pc, BH_DISCARD, f_zero);
        inject(bhir, ++pc, BH_IDENTITY, output, f_abs);
        inject(bhir, ++pc, BH_FREE, f_abs);
        inject(bhir, ++pc, BH_DISCARD, f_abs);
        inject(bhir, ++pc, BH_DIVIDE, output, input, output);
    }

    return pc-start_pc;
}

}}}
