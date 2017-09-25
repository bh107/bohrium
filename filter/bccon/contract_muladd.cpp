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

#include <bh_component.hpp>

using namespace std;

namespace bohrium {
namespace filter {
namespace bccon {

static bool rewrite_chain(BhIR &bhir, const vector<bh_instruction*>& chain, const vector<bh_view*>& temps)
{
    bh_instruction& first  = *chain.at(0); // BH_MULTIPLY
    bh_instruction& second = *chain.at(1); // BH_MULTIPLY
    bh_instruction& third  = *chain.at(2); // BH_ADD or BH_SUBTRACT

    vector<bh_instruction*> frees;

    for(size_t pc = 0; pc < bhir.instr_list.size(); ++pc) {
        bh_instruction& instr = bhir.instr_list[pc];

        // Skip if the instruction is one of the three we are looking at
        if (instr == first or instr == second or instr == third) {
            continue;
        }

        for(auto it : temps) {
            bh_view view = *it;

            if (instr.opcode == BH_FREE) {
                if (view == instr.operand[0]) {
                    frees.push_back(&instr);
                }
            } else if (instr.opcode != BH_NONE) {
                if (view == instr.operand[0] or
                    view == instr.operand[1] or
                    view == instr.operand[2]) {
                    verbose_print("[Muladd] \tCan't rewrite - Found use of view in other place!");
                    return false;
                }
            }
        }
    }

    if (frees.size() != temps.size()) {
        verbose_print("[Muladd] \tCan't rewrite - Not same amount of views as frees!");
        return false;
    }

    if (third.opcode == BH_ADD) {
        first.constant.set_double(first.constant.get_double() + second.constant.get_double());
    } else { // BH_SUBTRACT
        first.constant.set_double(first.constant.get_double() - second.constant.get_double());
    }

    // Set the constant type to the result type
    first.constant.type = third.operand[0].base->type;

    // The result of the first operations should be that of the thrid
    first.operand[0] = third.operand[0];

    // Remove unnecessary BH_FREE
    for (auto it : frees) {
        it->opcode = BH_NONE;
    }

    // Set the two other operations to BH_NONE
    second.opcode = BH_NONE;
    third.opcode  = BH_NONE;

    return true;
}
/*
We are looking for sequences like:

  BH_MULTIPLY a1 2 a0
  BH_MULTIPLY a2 3 a0
  BH_ADD a3 a1 a2

which might arise from the following math:

  2x + 3x

this can obviously be rewritten as:

  5x

or in the case of our byte-code:

  BH_MULTIPLY a3 5 a0
*/

void Contracter::contract_muladd(BhIR &bhir)
{
    bool rewritten = false;

    vector<bh_view*> temp_results;
    vector<bh_instruction*> instruction_chain;

    for(size_t pc = 0; pc < bhir.instr_list.size(); ++pc) {
        if (rewritten) {
            // We might catch more rewrites if we found one
            // so we loop back to the beginning
            pc        = 0;
            rewritten = false;
            temp_results.clear();
            instruction_chain.clear();
        }

        bh_instruction& instr = bhir.instr_list[pc];
        bh_view* multiplying_view;

        if (instr.opcode == BH_MULTIPLY) {
            if (bh_is_constant(&(instr.operand[1]))) {
                multiplying_view = &(instr.operand[2]);
            } else if (bh_is_constant(&(instr.operand[2]))) {
                multiplying_view = &(instr.operand[1]);
            } else {
                continue;
            }

            instruction_chain.push_back(&instr); // First BH_MULTIPLY found
            temp_results.push_back(&(instr.operand[0]));

            for(size_t sub_pc = pc+1; sub_pc < bhir.instr_list.size(); ++sub_pc) {
                if (rewritten) break;

                bh_instruction& other_instr = bhir.instr_list[sub_pc];

                if (other_instr.opcode == BH_MULTIPLY) {
                    if (!((bh_is_constant(&(other_instr.operand[1])) and *multiplying_view == other_instr.operand[2]) or
                          (bh_is_constant(&(other_instr.operand[2])) and *multiplying_view == other_instr.operand[1]))) {
                        continue;
                    }

                    instruction_chain.push_back(&other_instr); // Second BH_MULTIPLY found
                    temp_results.push_back(&(other_instr.operand[0]));

                    for(size_t sub_sub_pc = sub_pc+1; sub_sub_pc < bhir.instr_list.size(); ++sub_sub_pc) {
                        if (rewritten) break;

                        bh_instruction& yet_another_instr = bhir.instr_list[sub_sub_pc];

                        if (yet_another_instr.opcode == BH_ADD or yet_another_instr.opcode == BH_SUBTRACT) {
                            uint found = 0;
                            for(auto it : temp_results) {
                                if (*it == yet_another_instr.operand[1] or *it == yet_another_instr.operand[2]) {
                                    found += 1;
                                }
                            }

                            if (found >= 2) {
                                instruction_chain.push_back(&yet_another_instr);
                                verbose_print("[Muladd] Rewriting chain of length " + std::to_string(instruction_chain.size()));
                                rewritten = rewrite_chain(bhir, instruction_chain, temp_results);
                            }
                        }
                    }

                    instruction_chain.pop_back();
                    temp_results.pop_back();
                }
            }
        }
    }
}

}}}
