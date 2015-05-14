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
    composite.opcode = BH_NONE; // Easy choice... no re-use just NOOP it.

    bh_view result  = composite.operand[0];         // Grab operands
    bh_view input   = composite.operand[1];

    bh_view t1 = result;                            // Construct temps
    t1.base = make_base(BH_BOOL, result.base->nelem);
    bh_view t2 = result;
    t2.base = make_base(BH_BOOL, result.base->nelem);
    bh_view t3 = result;
    t3.base = make_base(BH_INT8, result.base->nelem);
    bh_view t4 = result;
    t4.base = make_base(BH_BOOL, result.base->nelem);
    bh_view t5 = result;
    t5.base = make_base(BH_BOOL, result.base->nelem);

    inject(bhir, ++pc, BH_LESS, t1, input, 0.0);    // Expand sequence
    inject(bhir, ++pc, BH_GREATER, t2, input, 0.0);
    inject(bhir, ++pc, BH_SUBTRACT, t3, t1, t2);
    inject(bhir, ++pc, BH_IDENTITY, result, t3);
    inject(bhir, ++pc, BH_FREE, t1);
    inject(bhir, ++pc, BH_DISCARD, t1);
    inject(bhir, ++pc, BH_FREE, t2);
    inject(bhir, ++pc, BH_DISCARD, t2);
    inject(bhir, ++pc, BH_FREE, t3);
    inject(bhir, ++pc, BH_DISCARD, t3);

    return pc-start_pc;
}

}}}
