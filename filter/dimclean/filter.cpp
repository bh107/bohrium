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
#include <bh_component.h>
#include <bh.h>

void filter(bh_ir &bhir)
{
    for (bh_instruction& instr: bhir.instr_list)
    {
        const int nop = bh_noperands(instr.opcode);
        bool sweep = bh_opcode_is_sweep(instr.opcode);
        for(int o=0; o<nop; ++o)
        {
            bh_view& view = instr.operand[o];
            if (sweep && o == 1 && view.shape[instr.constant.value.int64] == 1)
            {
                instr.opcode = BH_IDENTITY;
                sweep = false;
            }
            if (view.base) // Not a constant
            {
                bh_intp ndim = view.ndim;
                int ii = 0;
                for (int i = 0; i < ndim; ++i)
                {
                    if (view.shape[i] == 0)
                    {
                        instr.opcode = BH_NONE;
                    }
                    else if (view.shape[i] == 1)
                    {
                        if (view.ndim > 1)
                            --view.ndim;
                        if (sweep && o == 1 && instr.constant.value.int64 > ii)
                            --instr.constant.value.int64;
                    }
                    else
                    {
                        if (ii<i)
                        {
                            view.shape[ii] = view.shape[i];
                            view.stride[ii++] = view.stride[i];
                        }
                        else
                            ++ii;
                    }
                }
            }
        }
    }
}
