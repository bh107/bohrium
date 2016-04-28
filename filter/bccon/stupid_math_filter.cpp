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

using namespace std;

bool bh_constant_is_value(bh_constant constant, float_t value)
{
    switch(constant.type) {
        case BH_UINT8:
            return constant.value.uint8 == (uint8_t)value;
        case BH_UINT16:
            return constant.value.uint16 == (uint16_t)value;
        case BH_UINT32:
            return constant.value.uint32 == (uint32_t)value;
        case BH_UINT64:
            return constant.value.uint64 == (uint64_t)value;

        case BH_INT8:
            return constant.value.int8 == (int8_t)value;
        case BH_INT16:
            return constant.value.int16 == (int16_t)value;
        case BH_INT32:
            return constant.value.int32 == (int32_t)value;
        case BH_INT64:
            return constant.value.int64 == (int64_t)value;

        // We can't use floating point, since we'll lose precision
        /*
        case BH_FLOAT32:
            return constant.value.float32 == value;
        case BH_FLOAT64:
            return constant.value.float64 == value;
        */

        default:
            return false;
    }
}

bool is_multiplying_by_one(bh_instruction& instr)
{
    return instr.opcode == BH_MULTIPLY and
           bh_is_constant(&(instr.operand[2])) and
           bh_constant_is_value(instr.constant, 1.0);
}

bool is_dividing_by_one(bh_instruction& instr)
{
    return instr.opcode == BH_DIVIDE and
           bh_is_constant(&(instr.operand[2])) and
           bh_constant_is_value(instr.constant, 1.0);
}

bool is_adding_zero(bh_instruction& instr)
{
    return instr.opcode == BH_ADD and
           bh_is_constant(&(instr.operand[2])) and
           bh_constant_is_value(instr.constant, 0.0);
}

bool is_subtracting_zero(bh_instruction& instr)
{
    return instr.opcode == BH_SUBTRACT and
           bh_is_constant(&(instr.operand[2])) and
           bh_constant_is_value(instr.constant, 0.0);
}

bool is_free_or_discard(bh_instruction& instr)
{
    return instr.opcode == BH_FREE or
           instr.opcode == BH_DISCARD;
}

bool is_doing_stupid_math(bh_instruction& instr)
{
    return is_multiplying_by_one(instr) or
           is_dividing_by_one(instr) or
           is_adding_zero(instr) or
           is_subtracting_zero(instr);
}

void stupid_math_filter(bh_ir &bhir)
{
    for(size_t pc = 0; pc < bhir.instr_list.size(); ++pc) {
        bh_instruction& instr = bhir.instr_list[pc];

        if (is_doing_stupid_math(instr)) {
            // We could have the following:
            //   BH_MULTIPLY B A 0
            //   ...
            //   BH_FREE A
            //   BH_DISCARD A

            bh_base* B = instr.operand[0].base;
            bh_base* A = instr.operand[1].base;

            bool freed     = false;
            bool discarded = false;

            for (size_t pc_chain = pc+1; pc_chain < bhir.instr_list.size(); ++pc_chain) {
                bh_instruction& other_instr = bhir.instr_list[pc_chain];

                // Look for matching FREE and DISCARD for B
                if (is_free_or_discard(other_instr) and (other_instr.operand[0].base == B)) {
                    freed     = freed     or other_instr.opcode == BH_FREE;
                    discarded = discarded or other_instr.opcode == BH_DISCARD;
                }
            }

            // Check that B is created by us, that is, it isn't created prior to this stupid math call.
            bool created_before = false;
            for (size_t pc_chain = 0; pc_chain < pc; ++pc_chain) {
                bh_instruction& other_instr = bhir.instr_list[pc_chain];
                for (int idx = 0; idx < bh_noperands(other_instr.opcode); ++idx) {
                    created_before = created_before or other_instr.operand[idx].base == B;
                }
                if (created_before) break;
            }

            // Only if we FREE and DISCARD B in the same flush, are we allowed to change things.
            if (freed and discarded and !created_before) {
                for (size_t pc_chain = pc+1; pc_chain < bhir.instr_list.size(); ++pc_chain) {
                    bh_instruction& other_instr = bhir.instr_list[pc_chain];

                    // Look for matching FREE and DISCARD for A
                    if (is_free_or_discard(other_instr) and (other_instr.operand[0].base == A)) {
                        other_instr.opcode = BH_NONE; // Remove instruction
                    }

                    // Rewrite all uses of B to A
                    for (int idx = 0; idx < bh_noperands(other_instr.opcode); ++idx) {
                        if (other_instr.operand[idx].base == B) {
                            other_instr.operand[idx].base = A;
                        }
                    }
                }

                // Remove self
                instr.opcode = BH_NONE;
            }
        }
    }
}
