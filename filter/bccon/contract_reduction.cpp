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
#include "contracter.hpp"

using namespace std;

namespace bohrium {
namespace filter {
namespace bccon {

static void rewrite_chain(vector<bh_instruction*>& links, bh_instruction* &first, bh_instruction* &last)
{
    // Rewrite the first reduction as a "COMPLETE" REDUCE.
    // Copy the meta-data of the SCALAR output from the last REDUCE
    first->operand[0] = last->operand[0];

    // Set the last reduction to NONE, it no longer needs execution.
    last->opcode = BH_NONE;

    // Set all the instructions "links" in the chain as BH_NONE
    // they no longer need execution.
    vector<bh_instruction*>::iterator rit;
    for(rit = links.begin(); rit != links.end(); ++rit) {
        bh_instruction& rinstr = **rit;
        rinstr.opcode = BH_NONE;
    }
}

void Contracter::contract_reduction(BhIR &bhir)
{
    bh_opcode reduce_opcode = BH_NONE;
    bh_instruction* first;
    bh_instruction* last;

    std::set<const bh_base*> bases;

    // Instructions in the chain that are not, the first and the last reduction.
    vector<bh_instruction*> links;

    for(size_t pc = 0; pc < bhir.instr_list.size(); ++pc) {
        bh_instruction& instr = bhir.instr_list[pc];

        // Look for the "first" reduction in a chain of reductions
        if (bh_opcode_is_reduction(instr.opcode) and instr.operand[0].base->nelem > 1) {
            reduce_opcode = instr.opcode;
            bases.insert(instr.operand[0].base);

            first = &instr;
            last  = NULL;

            for(size_t pc_chain = pc+1; pc_chain < bhir.instr_list.size(); ++pc_chain) {
                bh_instruction& other_instr = bhir.instr_list[pc_chain];

                bool other_use = false;
                for(bh_view &other_view: other_instr.operand) {
                    if (bh_is_constant(&other_view)) {
                        continue;
                    }

                    if (bases.find(other_view.base) != bases.end()) {
                        other_use = true;
                        break;
                    }
                }

                if (!other_use) {
                    continue;
                }

                bool is_none    = other_instr.opcode == BH_NONE;
                bool is_freed   = other_instr.opcode == BH_FREE;
                bool is_reduced = other_instr.opcode == reduce_opcode;

                if (!(is_none or is_freed or is_reduced)) {
                    // Chain is broken - Reset the search
                    first = NULL;
                    break;
                } else if (other_instr.operand[0].base->nelem == 1) {
                    // Scalar output - End of the chain
                    last = &other_instr;
                } else {
                    links.push_back(&other_instr);
                    if (other_instr.opcode == reduce_opcode) {
                        bases.insert(other_instr.operand[0].base);
                    }
                }
            }

            if (first and last) {
                verbose_print("[Reduction] Rewriting chain of length " + std::to_string(links.size()));
                rewrite_chain(links, first, last);
            }

            // Reset the search
            reduce_opcode = BH_NONE;
            first         = NULL;
            last          = NULL;
            links.clear();
            bases.clear();
        }
    }
}

}}}
