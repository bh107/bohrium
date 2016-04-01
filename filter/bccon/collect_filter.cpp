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

float_t bh_get_value(bh_constant constant)
{
    switch(constant.type) {
        case BH_UINT8:
            return (float_t)constant.value.uint8;
        case BH_UINT16:
            return (float_t)constant.value.uint16;
        case BH_UINT32:
            return (float_t)constant.value.uint32;
        case BH_UINT64:
            return (float_t)constant.value.uint64;

        case BH_INT8:
            return (float_t)constant.value.int8;
        case BH_INT16:
            return (float_t)constant.value.int16;
        case BH_INT32:
            return (float_t)constant.value.int32;
        case BH_INT64:
            return (float_t)constant.value.int64;

        case BH_FLOAT32:
            return (float_t)constant.value.float32;
        case BH_FLOAT64:
            return (float_t)constant.value.float64;

        default:
            fprintf(stderr, "Don't know this type (%s) for collect filter.\n", bh_type_text(constant.type));
            return 0;
    }
}

void bh_set_value(bh_constant* constant, float_t value)
{
    switch(constant->type) {
        case BH_UINT8:
            constant->value.uint8 = (uint)value;
            break;
        case BH_UINT16:
            constant->value.uint16 = (uint)value;
            break;
        case BH_UINT32:
            constant->value.uint32 = (uint)value;
            break;
        case BH_UINT64:
            constant->value.uint64 = (uint)value;
            break;

        case BH_INT8:
            constant->value.int8 = (int)value;
            break;
        case BH_INT16:
            constant->value.int16 = (int)value;
            break;
        case BH_INT32:
            constant->value.int32 = (int)value;
            break;
        case BH_INT64:
            constant->value.int64 = (int)value;
            break;

        case BH_FLOAT32:
            constant->value.float32 = value;
            break;
        case BH_FLOAT64:
            constant->value.float64 = value;
            break;

        default:
            fprintf(stderr, "Can't set value for this type (%s) for collect filter.\n", bh_type_text(constant->type));
    }
}

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
            return;
    }

    float_t sum = 0.0;

    // Update first instruction's result base to last
    first.operand[0].base = last.operand[0].base;

    // Get first instructions value
    if (first.opcode == BH_ADD) {
        sum += bh_get_value(first.constant);
    } else {
        sum -= bh_get_value(first.constant);
    }

    // Loop through rest and accumulate value
    for(vector<bh_instruction*>::iterator ite=chain.begin()+1; ite != chain.end(); ++ite) {
        bh_instruction& rinstr = **ite;
        if (rinstr.opcode == BH_ADD) {
            sum += bh_get_value(rinstr.constant);
        } else {
            sum -= bh_get_value(rinstr.constant);
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
    bh_set_value(&(first.constant), sum);
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
            return;
    }

    float_t result = 1.0;

    // Update first instruction's result base to last
    first.operand[0].base = last.operand[0].base;

    // Get first instructions value
    if (first.opcode == BH_MULTIPLY) {
        result *= bh_get_value(first.constant);
    } else {
        result /= bh_get_value(first.constant);
    }

    // Loop through rest and accumulate value
    for(vector<bh_instruction*>::iterator ite=chain.begin()+1; ite != chain.end(); ++ite) {
        bh_instruction& rinstr = **ite;
        if (rinstr.opcode == BH_MULTIPLY) {
            result *= bh_get_value(rinstr.constant);
        } else {
            result /= bh_get_value(rinstr.constant);
        }
        // Remove instruction
        rinstr.opcode = BH_NONE;
    }

    // Set first instruction's new value
    bh_set_value(&(first.constant), result);
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
                        if (chain.size() > 1) {
                            rewrite_chain(chain);
                        }
                        // Reset
                        chain.clear();
                        views.clear();
                        break;
                    }
                }
            }
        }

        chain.clear();
        views.clear();
    }
}
