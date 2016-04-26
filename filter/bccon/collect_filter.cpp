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

bool is_add_sub(bh_opcode opc)
{
    return opc == BH_ADD or opc == BH_SUBTRACT;
}

bool is_mul_div(bh_opcode opc)
{
    return opc == BH_MULTIPLY or opc == BH_DIVIDE;
}

bool chain_has_same_type(vector<bh_instruction*>& chain)
{
    bh_type type = chain.front()->constant.type;
    for(vector<bh_instruction*>::iterator ite=chain.begin()+1; ite != chain.end(); ++ite) {
        if (type != (**ite).constant.type)
            return false;
    }
    return true;
}

void rewrite_chain_add_sub(vector<bh_instruction*>& chain)
{
    bh_instruction& first = *chain.front();
    bh_instruction& last = *chain.back();

    if (!chain_has_same_type(chain))
        return;

    switch (first.constant.type) {
        // Don't know how to do complex types, yet.
        case BH_BOOL:
        case BH_COMPLEX64:
        case BH_COMPLEX128:
        case BH_R123:
            return;
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

void rewrite_chain_mul_div(vector<bh_instruction*>& chain)
{
    bh_instruction& first = *chain.front();
    bh_instruction& last = *chain.back();

    if (!chain_has_same_type(chain))
        return;

    switch (first.constant.type) {
        // Don't know how to do complex types, yet.
        case BH_BOOL:
        case BH_COMPLEX64:
        case BH_COMPLEX128:
        case BH_R123:
            return;
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
    first.constant.set_double(result);
}

void rewrite_chain(vector<bh_instruction*>& chain)
{
    bh_opcode opc = chain[0]->opcode;
    if (is_add_sub(opc)) {
        rewrite_chain_add_sub(chain);
    } else if (is_mul_div(opc)) {
        rewrite_chain_mul_div(chain);
    }
}

void collect_filter(bh_ir &bhir)
{
    bh_opcode collect_opcode = BH_NONE;
    vector<bh_view*> views;
    vector<bh_instruction*> chain;

    for(size_t pc = 0; pc < bhir.instr_list.size(); ++pc) {
        bh_instruction& instr = bhir.instr_list[pc];

        if ((is_add_sub(instr.opcode) or is_mul_div(instr.opcode)) and bh_is_constant(&(instr.operand[2]))) {
            collect_opcode = instr.opcode;
            views.push_back(&instr.operand[0]);
            chain.push_back(&instr);

            for(size_t pc_chain = pc+1; pc_chain < bhir.instr_list.size(); ++pc_chain) {
                bh_instruction& other_instr = bhir.instr_list[pc_chain];

                if (is_add_sub(collect_opcode) and is_add_sub(other_instr.opcode) and bh_is_constant(&instr.operand[2])) {
                    // Both are ADD or SUBTRACT
                    if (*views.back() == other_instr.operand[1]) {
                        views.push_back(&other_instr.operand[0]);
                        chain.push_back(&other_instr);
                    }
                } else if (is_mul_div(collect_opcode) and is_mul_div(other_instr.opcode) and bh_is_constant(&instr.operand[2])) {
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
                    bool is_none      = other_instr.opcode == BH_NONE;
                    bool is_freed     = other_instr.opcode == BH_FREE;
                    bool is_discarded = other_instr.opcode == BH_DISCARD;

                    if (is_none or is_freed or is_discarded) {
                        continue;
                    } else {
                        // Is not ADD, SUBTRACT, MULTIPLY, DIVIDE, NONE, FREE, DISCARD
                        // End chain
                        if (chain.size() > 1)
                            rewrite_chain(chain);

                        // Reset
                        chain.clear();
                        views.clear();
                        break;
                    }
                }
            }
        }

        // Rewrite if end of instruction list
        if (chain.size() > 1)
            rewrite_chain(chain);

        chain.clear();
        views.clear();
    }
}
