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
#include <bh.h>
#include <stdio.h>

using namespace std;

typedef vector<bh_instruction> ilist;
typedef ilist::iterator ilist_iter;

void rewrite_chain(vector<bh_instruction*>& links, bh_instruction* first, bh_instruction* last)
{
    // Rewrite the first reduction as a "COMPLETE" REDUCE.
    // Copy the meta-data of the SCALAR output from the last REDUCE
    first->operand[0] = last->operand[0];

    // Set the last reduction to NONE, it no longer needs execution.
    last->opcode = BH_NONE;

    // Set all the instructions "links" in the chain as BH_NONE
    // they no longer need execution.
    for(vector<bh_instruction*>::iterator rit=links.begin();
        rit!=links.end();
        ++rit) {
        bh_instruction& rinstr = **rit;
        rinstr.opcode = BH_NONE;
    }
}

void filter(bh_ir &bhir)
{
    bh_base* reduce_output = NULL;
    bh_opcode reduce_opcode = BH_NONE;
    bh_instruction* first;
    bh_instruction* last;

    vector<bh_instruction*> links;  // Instructions in the chain that are not,
                                    // the first and the last reduction.

    for(ilist_iter it = bhir.instr_list.begin();
        it!=bhir.instr_list.end();
        ++it) {
        bh_instruction& instr = *it;

        // Find the "first" reduction in a reduction chain.
        if ((reduce_output == NULL) and \
            (bh_opcode_is_reduction(instr.opcode))) {

            reduce_output = instr.operand[0].base;
            reduce_opcode = instr.opcode;

            first = &instr;

            printf("Beginning the chain...\n");

        // A potential continuation of the chain
        } else if ( (reduce_output != NULL) and \
                    (reduce_opcode == instr.opcode) and \
                    (reduce_output == instr.operand[1].base)) {

            bool other_use=false, gets_freed=false, gets_discarded=false;
            for(ilist_iter rit(it+1); rit!=bhir.instr_list.end(); ++rit) {
                bh_instruction& other_instr = *rit;
                switch(other_instr.opcode) {
                    case BH_FREE:
                        if (other_instr.operand[0].base == reduce_output) {
                            gets_freed = true;
                            links.push_back(&other_instr);
                        }
                        break;
                    case BH_DISCARD:
                        if (other_instr.operand[0].base == reduce_output) {
                            gets_discarded = true;

                            links.push_back(&other_instr);
                        }
                        break;
                    default:
                        for(int oidx=0; oidx<bh_operands(other_instr.opcode); ++oidx) {
                            if (bh_is_constant(&other_instr.operand[oidx])) {
                                continue;
                            }
                            if (other_instr.operand[oidx].base == reduce_output) {
                                other_use = true;
                            }
                        }
                        break;
                }
                // Can stop looking further if it gets used by something else
                // or if it gets freed and discarded.
                if (other_use or (gets_freed and gets_discarded)) {
                    break;
                }
            }
            
            bool is_continuation = gets_freed and gets_discarded and not other_use;
            bool is_scalar = (instr.operand[0].ndim == 1) and (instr.operand[0].shape[0] == 1);

            reduce_output = instr.operand[0].base;

            if (is_continuation and is_scalar) {            // End of the chain

                printf("Ending the chain and REWRITE as COMPLETE REDUCE\n");
                last = &instr;

                rewrite_chain(links, first, last);

            } else if (is_continuation and not is_scalar) { // Continuation
                printf("Continuing the chain...\n");
                links.push_back(&instr);
            } else {                                        // Break the chain.
                printf("Break the chain.\n");
            }

            if (not is_continuation or is_scalar) {         // Reset the search
                printf("Resetting search.\n");
                reduce_output = NULL;
                reduce_opcode = BH_NONE;
                links.clear();
            }

        // A break
        } else if (reduce_output != NULL) {
            reduce_output = NULL;
            reduce_opcode = BH_NONE;
            links.clear();
            printf("Breaking the chain...\n");
        }
    }
}
