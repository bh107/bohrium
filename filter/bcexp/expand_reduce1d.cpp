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
namespace bcexp {

static std::map<int, int> fold_map;

static inline int find_fold(int64_t elements, int thread_limit)
{
    for (int i = elements/thread_limit; i > 1; i--)
    {
        if (elements%i == 0)
            return i;
    }
    return 1;
}

// TODO: What does this do? Give example like expand_sign.cpp
int Expander::expand_reduce1d(BhIR& bhir, int pc, int thread_limit)
{
    int start_pc = pc;
    bh_instruction& instr = bhir.instr_list[pc];
    bh_opcode opcode = instr.opcode;
    int64_t elements = bh_nelements(instr.operand[1]);
    verbose_print("[Reduce1D] Expanding " + string(bh_opcode_text(opcode)));

    if (elements * 2 < thread_limit) {
        return 0;
    }

    int fold = 0;
    if (fold_map.find(elements) != fold_map.end()) {
        fold = fold_map.find(elements)->second;
    } else {
        fold = find_fold(elements,thread_limit);
        fold_map[elements] = fold;
    }

    if (fold < 2) {
        verbose_print("[Reduce1D] \tCan't expand " + string(bh_opcode_text(opcode)) + " with a fold less than 2.");
        return 0;
    }

    // Lazy choice... no re-use just NOP it.
    instr.opcode = BH_NONE;

    // Grab operands
    bh_view out = instr.operand[0];
    bh_view in  = instr.operand[1];

    in.ndim = 2;
    in.shape[0] = fold;
    in.shape[1] = elements / fold;

    in.stride[1] = in.stride[0];
    in.stride[0] = in.stride[0] * elements / fold;

    bh_view temp = make_temp(in.base->type, elements/fold);

    inject(bhir, ++pc, opcode,  temp, in,   0, bh_type::INT64);
    inject(bhir, ++pc, opcode,  out,  temp, 0, bh_type::INT64);
    inject(bhir, ++pc, BH_FREE, temp);

    return pc - start_pc;
}

}}}
