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
#include <stdio.h>
#include <set>

#include <bh_component.h>
#include <bh.h>

using namespace std;

typedef vector<bh_instruction> ilist;
typedef ilist::iterator ilist_iter;

/*
    The implementation currently only detects chains of reductions as produced by
    np.sum(), it does not detect chains created by np.add.reduce(np.add.reduce()).
    This could and should be remedied.

    With the input::

    "np.sum(np.ones((10,10,10)))" produces::

    ADD_REDUCE(t1, a)
    ADD_REDUCE(t2, t1)
    ADD_REDUCE(s, t2)
    BH_FREE(t2)
    BH_DISCARD(t2)
    BH_FREE(t1)
    BH_DISCARD(t1)
    BH_FREE(a)
    BH_DISCARD(a)

    Which the implementation handles.

    np.add.reduce(np.add.reduce(np.add.reduce(np.ones((10,10,10))))) produces::

    ADD_REDUCE(t1, a)
    BH_FREE(a)
    BH_DISCARD(a)
    ADD_REDUCE(t2, t1)
    BH_FREE(t1)
    BH_DISCARD(t1)
    ADD_REDUCE(s, t2)
    BH_FREE(t2)
    BH_DISCARD(t2)

    There are other permutations of reduce-chains that are not detected.
    However, the above is sufficient to start using/implementing/testing
    complete reductions.
*/

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
    bh_opcode reduce_opcode = BH_NONE;
    bh_instruction* first;
    bh_instruction* last;

    std::set<bh_base*> bases;

    vector<bh_instruction*> links;  // Instructions in the chain that are not,
                                    // the first and the last reduction.

    for(size_t pc=0; pc<bhir.instr_list.size(); ++pc) {
        bh_instruction& instr = bhir.instr_list[pc];

        // Look for the "first" reduction in a chain of reductions
        if (bh_opcode_is_reduction(instr.opcode) and (instr.operand[0].base->nelem > 1)) {

            reduce_opcode = instr.opcode;
            bases.insert(instr.operand[0].base);

            first = &instr;
            last = NULL;

            //printf("Beginning the chain...\n");
            //bh_pprint_instr(&instr);

            for(size_t pc_chain=pc+1; pc_chain<bhir.instr_list.size(); ++pc_chain) {

                bh_instruction& other_instr = bhir.instr_list[pc_chain];

                bool other_use=false;                   // Check for other use
                for(int oidx=0; oidx < bh_noperands(other_instr.opcode); ++oidx) {
                    if (bh_is_constant(&other_instr.operand[oidx])) {
                        continue;
                    }
                    if (bases.find(other_instr.operand[oidx].base) != bases.end()) {
                        other_use = true;
                    }
                }
                if (!other_use) {                       // Ignore it
                    //printf("IGNORING\n");
                    //bh_pprint_instr(&other_instr);
                    continue;
                }

                int gets_freed = other_instr.opcode == BH_FREE;
                int gets_discarded = other_instr.opcode == BH_DISCARD;
                int gets_reduced = other_instr.opcode == reduce_opcode;
                bool is_continuation = gets_freed or gets_discarded or gets_reduced;
                bool scalar_output = (other_instr.operand[0].base->nelem == 1);

                if (not is_continuation) {              // Chain is broken
                    //printf("CHAIN BROKEN BY: \n");
                    //bh_pprint_instr(&other_instr);

                    reduce_opcode = BH_NONE;            // Reset the search
                    first = NULL;
                    last = NULL;
                    links.clear();
                    bases.clear();
                }

                if (scalar_output) {                    // End of the chain
                    //printf("Ending the chain and REWRITE as COMPLETE REDUCE\n");
                    //bh_pprint_instr(&other_instr);

                    last = &other_instr;
                } else {                                // Continuation
                    //printf("Continuing the chain...\n");
                    //bh_pprint_instr(&other_instr);

                    links.push_back(&other_instr);
                    if (other_instr.opcode == reduce_opcode) {    // Update reduce_output
                        bases.insert(other_instr.operand[0].base);
                    }
                }
            }

            if (first and last) {                       // Rewrite the chain
                rewrite_chain(links, first, last);
            }
            reduce_opcode = BH_NONE;                    // Reset the search
            first = NULL;
            last = NULL;
            links.clear();
            bases.clear();
        }
    }
}
