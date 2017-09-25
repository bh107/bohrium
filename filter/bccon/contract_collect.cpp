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

static inline bool is_add_sub(const bh_opcode& opc)
{
    return opc == BH_ADD or opc == BH_SUBTRACT;
}

static inline bool is_mul_div(const bh_opcode& opc)
{
    return opc == BH_MULTIPLY or opc == BH_DIVIDE;
}

static inline bool is_none_free(const bh_opcode& opc)
{
    return opc == BH_NONE or opc == BH_FREE;
}

static bool chain_has_same_type(vector<bh_instruction*>& chain)
{
    const bh_type type = chain.front()->constant.type;

    for(auto const instr : chain) {
        if (type != instr->constant.type) {
            return false;
        }
    }

    return true;
}

static void rewrite_chain_add_sub(BhIR &bhir, vector<bh_instruction*>& chain)
{
    bh_instruction& first = *chain.front();
    bh_instruction& last = *chain.back();

    if (!chain_has_same_type(chain)) {
        verbose_print("[Collect] \tAddsub chain doesn't have same type.");
        return;
    }

    switch (first.constant.type) {
        case bh_type::BOOL:
        case bh_type::COMPLEX64:
        case bh_type::COMPLEX128:
        case bh_type::R123:
            verbose_print("[Collect] \tDon't know how to do complex types, yet.");
            return;
        default:
            break;
    }

    float_t sum = 0.0;

    // Update first instruction's result base to last
    first.operand[0].base = last.operand[0].base;

    // Get first instructions value
    if (first.opcode == BH_ADD) {
        sum += first.constant.get_double();
    } else {
        sum -= first.constant.get_double();
    }

    // Loop through rest and accumulate value
    for(vector<bh_instruction*>::iterator ite=chain.begin()+1; ite != chain.end(); ++ite) {
        bh_instruction& rinstr = **ite;
        if (rinstr.opcode == BH_ADD) {
            sum += rinstr.constant.get_double();
        } else {
            sum -= rinstr.constant.get_double();
        }
        // Remove instruction
        rinstr.opcode = BH_NONE;
    }

    // We might have to reverse the original first opcode
    // If sum is below zero, we want to subtract
    if (sum < 0) {
        first.opcode = BH_SUBTRACT;
        sum = -sum;
    } else {
        first.opcode = BH_ADD;
    }

    // Set first instruction's new value
    first.constant.set_double(sum);
}

static void rewrite_chain_mul_div(BhIR &bhir, vector<bh_instruction*>& chain)
{
    bh_instruction& first = *chain.front();
    bh_instruction& last = *chain.back();

    if (!chain_has_same_type(chain)) {
        verbose_print("[Collect] \tMuldiv chain doesn't have same type.");
        return;
    }

    switch (first.constant.type) {
        case bh_type::BOOL:
        case bh_type::COMPLEX64:
        case bh_type::COMPLEX128:
        case bh_type::R123:
            verbose_print("[Collect] \tDon't know how to do complex types, yet.");
            return;
        default:
            break;
    }

    float_t result = 1.0;

    // Update first instruction's result base to last
    first.operand[0].base = last.operand[0].base;

    // Get first instructions value
    if (first.opcode == BH_MULTIPLY) {
        result *= first.constant.get_double();
    } else {
        result /= first.constant.get_double();
    }

    // Loop through rest and accumulate value
    for(vector<bh_instruction*>::iterator ite=chain.begin()+1; ite != chain.end(); ++ite) {
        bh_instruction& rinstr = **ite;
        if (rinstr.opcode == BH_MULTIPLY) {
            result *= rinstr.constant.get_double();
        } else {
            result /= rinstr.constant.get_double();
        }
        // Remove instruction
        rinstr.opcode = BH_NONE;
    }

    // Set first instruction's new value
    first.opcode = BH_MULTIPLY;
    first.constant.set_double(result);
}

static void rewrite_chain(BhIR &bhir, vector<bh_instruction*>& chain)
{
    bh_opcode opc = chain[0]->opcode;
    if (is_add_sub(opc)) {
        verbose_print("[Collect] \tAddSub rewrite.");
        rewrite_chain_add_sub(bhir, chain);
    } else if (is_mul_div(opc)) {
        verbose_print("[Collect] \tMulDiv rewrite.");
        rewrite_chain_mul_div(bhir, chain);
    }
}

void Contracter::contract_collect(BhIR &bhir)
{
    bh_opcode collect_opcode = BH_NONE;
    vector<const bh_view*> views;
    vector<bh_instruction*> chain;

    for(size_t pc = 0; pc < bhir.instr_list.size(); ++pc) {
        bh_instruction& instr = bhir.instr_list[pc];

        if ((is_add_sub(instr.opcode) or is_mul_div(instr.opcode)) and bh_is_constant(&(instr.operand[2]))) {
            collect_opcode = instr.opcode;
            views.push_back(&instr.operand[0]);
            chain.push_back(&instr);

            for(size_t pc_chain = pc+1; pc_chain < bhir.instr_list.size(); ++pc_chain) {
                bh_instruction& other_instr = bhir.instr_list[pc_chain];

                if (is_add_sub(collect_opcode) and is_add_sub(other_instr.opcode) and bh_is_constant(&other_instr.operand[2])) {
                    // Both are ADD or SUBTRACT
                    if (*views.back() == other_instr.operand[1]) {
                        views.push_back(&other_instr.operand[0]);
                        chain.push_back(&other_instr);
                    }
                } else if (is_mul_div(collect_opcode) and is_mul_div(other_instr.opcode) and bh_is_constant(&other_instr.operand[2])) {
                    // Both are MULTIPLY or DIVIDE

                    // We are not allowed to DIVIDE when the result operand has integer type
                    if (bh_type_is_integer(other_instr.operand[0].base->type)) {
                        chain.clear();
                        views.clear();
                        break;
                    } else if (*views.back() == other_instr.operand[1]) {
                        views.push_back(&other_instr.operand[0]);
                        chain.push_back(&other_instr);
                    }
                } else {
                    if (is_none_free(other_instr.opcode)) {
                        continue;
                    } else {
                        // Is not ADD, SUBTRACT, MULTIPLY, DIVIDE, NONE, FREE
                        // End chain
                        if (chain.size() > 1) {
                            verbose_print("[Collect] Rewriting chain of length " + std::to_string(chain.size()));
                            rewrite_chain(bhir, chain);
                        }

                        // Reset
                        chain.clear();
                        views.clear();
                        break;
                    }
                }
            }
        }

        // Rewrite if end of instruction list
        if (chain.size() > 1) {
            verbose_print("[Collect] End of loop rewriting chain of length " + std::to_string(chain.size()));
            rewrite_chain(bhir, chain);
        }

        chain.clear();
        views.clear();
    }
}

}}}
