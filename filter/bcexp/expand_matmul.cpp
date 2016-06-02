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
 *  Expand matmul at the given pc in the bhir instruction list.
 *
 *  Returns the number of additional instructions used.
 */
int Expander::expand_matmul(bh_ir& bhir, int pc)
{
    int start_pc = pc;
    bh_instruction& composite = bhir.instr_list[pc];

    // Lazy choice... no re-use just NOP it.
    composite.opcode = BH_NONE;

    // Grab operands
    bh_view out = composite.operand[0];
    bh_view a   = composite.operand[1];
    bh_view b   = composite.operand[2];

    // Grab the shape
    int n = a.shape[0];
    int m = a.shape[1];
    int k = b.shape[1];

    // Construct intermediary operands
    // Needs broadcast
    bh_view a_3d = a;

    // Needs transposition + broadcast
    bh_view b_3d = b;

    a_3d.ndim = 3;
    b_3d.ndim = 3;

    // Set the shape
    a_3d.shape[0] = b_3d.shape[0] = n;
    a_3d.shape[1] = b_3d.shape[1] = k;
    a_3d.shape[2] = b_3d.shape[2] = m;

    // Construct broadcast
    a_3d.stride[0] = a.stride[0];
    a_3d.stride[1] = 0;
    a_3d.stride[2] = a.stride[1];

    // Construct transpose + broadcast
    b_3d.stride[0] = 0;
    b_3d.stride[1] = b.stride[1];
    b_3d.stride[2] = b.stride[0];

    // Construct temp for mul-result
    bh_view c_3d = b_3d;

    // Count number of elements
    bh_intp nelements = 1;

    // Set contiguous stride
    for(bh_intp dim=c_3d.ndim-1; dim >= 0; --dim) {
        c_3d.stride[dim] = nelements;
        nelements *= c_3d.shape[dim];
    }
    c_3d.start = 0;
    c_3d.base = make_base(b_3d.base->type, nelements);

    // Expand sequence
    inject(bhir, ++pc, BH_MULTIPLY,   c_3d, a_3d, b_3d);
    inject(bhir, ++pc, BH_ADD_REDUCE, out,  c_3d, (int64_t)2, BH_INT64);
    inject(bhir, ++pc, BH_FREE,       c_3d);

    verbose_print("[Matmul] Expanding BH_MATMUL");
    return pc - start_pc;
}

}}}
