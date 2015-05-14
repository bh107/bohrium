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

    bh_view out     = composite.operand[0];         // Grab operands
    bh_view input   = composite.operand[1];

    bh_type input_type = input.base->type;          // Grab the input-type

    bh_view meta = composite.operand[0];            // Inherit ndim and shape
    meta.start = 0;
    bh_intp nelements = 1;                          // Count number of elements
    for(bh_intp dim=meta.ndim-1; dim >= 0; --dim) { // Contiguous stride
        meta.stride[dim] = nelements;
        nelements *= meta.shape[dim];
    }
                                                            // NON-COMPLEX input-type
    if (!((input_type == BH_COMPLEX64) || (input_type == BH_COMPLEX128))) {
                                                            // Construct temps
        bh_view t1_bool = make_temp(meta, BH_BOOL, nelements);
        bh_view t1      = make_temp(meta, input_type, nelements);
        bh_view t2_bool = make_temp(meta, BH_BOOL, nelements);
        bh_view t2      = make_temp(meta, input_type, nelements);
        
        inject(bhir, ++pc, BH_LESS, t1_bool, input, 0.0);   // Expand sequence
        inject(bhir, ++pc, BH_IDENTITY, t1, t1_bool);
        inject(bhir, ++pc, BH_FREE, t1_bool);
        inject(bhir, ++pc, BH_DISCARD, t1_bool);

        inject(bhir, ++pc, BH_GREATER, t2_bool, input, 0.0);
        inject(bhir, ++pc, BH_IDENTITY, t2, t2_bool);
        inject(bhir, ++pc, BH_FREE, t2_bool);
        inject(bhir, ++pc, BH_DISCARD, t2_bool);

        inject(bhir, ++pc, BH_SUBTRACT, out, t2, t1);
        inject(bhir, ++pc, BH_FREE, t1);
        inject(bhir, ++pc, BH_DISCARD, t1);
        inject(bhir, ++pc, BH_FREE, t2);
        inject(bhir, ++pc, BH_DISCARD, t2);
    } else {                                                // COMPLEX input-type
        bh_type real_type = (input_type == BH_COMPLEX64) ? BH_FLOAT32 : BH_FLOAT64;

                                                            // Construct temps
        bh_view input_r = make_temp(meta, real_type, nelements);
        bh_view t1_bool = make_temp(meta, BH_BOOL, nelements); 
        bh_view t1      = make_temp(meta, real_type, nelements); 
        bh_view t2_bool = make_temp(meta, BH_BOOL, nelements);
        bh_view t2      = make_temp(meta, real_type, nelements); 
        bh_view t3      = make_temp(meta, real_type, nelements); 
        
        inject(bhir, ++pc, BH_REAL, input_r, input);
        
        inject(bhir, ++pc, BH_LESS, t1_bool, input_r, 0.0); // Expand sequence
        inject(bhir, ++pc, BH_IDENTITY, t1, t1_bool);
        inject(bhir, ++pc, BH_FREE, t1_bool);
        inject(bhir, ++pc, BH_DISCARD, t1_bool);

        inject(bhir, ++pc, BH_GREATER, t2_bool, input_r, 0.0);
        inject(bhir, ++pc, BH_FREE, input_r);
        inject(bhir, ++pc, BH_DISCARD, input_r);

        inject(bhir, ++pc, BH_IDENTITY, t2, t2_bool);
        inject(bhir, ++pc, BH_FREE, t2_bool);
        inject(bhir, ++pc, BH_DISCARD, t2_bool);

        inject(bhir, ++pc, BH_SUBTRACT, t3, t2, t1);
        inject(bhir, ++pc, BH_FREE, t1);
        inject(bhir, ++pc, BH_DISCARD, t1);
        inject(bhir, ++pc, BH_FREE, t2);
        inject(bhir, ++pc, BH_DISCARD, t2);

        inject(bhir, ++pc, BH_IDENTITY, out, t3);
        inject(bhir, ++pc, BH_FREE, t3);
        inject(bhir, ++pc, BH_DISCARD, t3);
    }

    return pc-start_pc;
}

}}}
