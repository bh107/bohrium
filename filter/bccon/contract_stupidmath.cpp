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

static inline bool is_multiplying_by_one(const bh_instruction& instr)
{
    return instr.opcode == BH_MULTIPLY and
           instr.constant.get_double() == 1.0;
}

static inline bool is_dividing_by_one(const bh_instruction& instr)
{
    return instr.opcode == BH_DIVIDE and
           instr.constant.get_double() == 1.0;
}

static inline bool is_adding_zero(const bh_instruction& instr)
{
    return instr.opcode == BH_ADD and
           instr.constant.get_double() == 0.0;
}

static inline bool is_subtracting_zero(const bh_instruction& instr)
{
    return instr.opcode == BH_SUBTRACT and
           instr.constant.get_double() == 0.0;
}

static inline bool is_entire_view(const bh_instruction& instr)
{
    for(const bh_view &view: instr.operand) {
        if (bh_is_contiguous(&view)) {
            return true;
        }
    }
    return false;
}

static inline bool is_doing_stupid_math(const bh_instruction& instr)
{
    return instr.has_constant() and
           bh_type_is_integer(instr.constant.type) and
           (
               is_multiplying_by_one(instr) or
               is_dividing_by_one(instr) or
               is_adding_zero(instr) or
               is_subtracting_zero(instr)
           ) and
           is_entire_view(instr);
}

void Contracter::contract_stupidmath(BhIR &bhir)
{
    for(bh_instruction& instr: bhir.instr_list) {
        if (is_doing_stupid_math(instr)) {
            verbose_print("[Stupid math] Is doing stupid math with a " + std::string(bh_opcode_text(instr.opcode)));

            // We could have the following:
            //   BH_ADD B A 0
            //   BH_FREE A
            //   BH_SYNC B
            // We want to find the add and replace it with BH_IDENTITY
            instr.opcode = BH_IDENTITY;

            // We need to figure out which operand is the constant, and remove it
            if (bh_is_constant(&(instr.operand[1]))) {
                instr.operand.erase(instr.operand.begin() + 1);
            } else {
                instr.operand.erase(instr.operand.begin() + 2);
            }
        }
    }
}

}}}
