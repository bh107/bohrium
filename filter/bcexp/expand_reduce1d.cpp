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
#include <map>
using namespace std;

namespace bohrium {
namespace filter {
namespace composite {

static inline int find_fold(bh_index elements, int thread_limit)
{
    for (int i = elements/thread_limit; i > 1; i--)
    {
        if (elements%i == 0)
            return i;
    }
    return 1;
}

int Expander::expand_reduce1d(bh_ir& bhir, int pc, int thread_limit)
{
    static std::map<int,int> fold_map;
    int start_pc = pc;                              
    bh_instruction& instr = bhir.instr_list[pc];        // Grab the BH_POWER instruction
    
    bh_index elements = bh_nelements(instr.operand[1]);

    if (elements * 2 < thread_limit)
        return 0;
        
    int fold = 0;
    if (fold_map.find(elements) != fold_map.end())
    {    
        fold = fold_map.find(elements)->second;
    } else {
        fold = find_fold(elements,thread_limit);
        fold_map[elements] = fold;
    }
    if (fold < 2)
        return 0;
    
    bh_opcode opcode = instr.opcode;
    instr.opcode = BH_NONE;             // Lazy choice... no re-use just NOP it.
    bh_view out = instr.operand[0];     // Grab operands
    bh_view in = instr.operand[1];
    in.ndim = 2;
    in.shape[0] = fold; 
    in.shape[1] = elements/fold;
    in.stride[1] = in.stride[0];
    in.stride[0] = in.stride[0]*elements/fold;
    bh_view temp = make_temp(in.base->type, elements/fold);
    inject(bhir, ++pc, opcode, temp, in, 0, BH_INT64);
    inject(bhir, ++pc, opcode, out, temp, 0, BH_INT64);
    inject(bhir, ++pc, BH_FREE, temp);
    inject(bhir, ++pc, BH_DISCARD, temp);

    return pc-start_pc;
}

}}}
