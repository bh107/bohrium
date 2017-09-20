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

#ifndef __BH_IR_H
#define __BH_IR_H

#include <vector>
#include <map>
#include <set>

#include <bh_instruction.hpp>

/* The Bohrium Internal Representation (BhIR) represents an instruction
 * batch created by the Bridge component typically. */
class bh_ir
{
public:
    //The list of Bohrium instructions in topological order
    std::vector<bh_instruction> instr_list;

public:
    // Some constructors
    bh_ir(std::vector<bh_instruction> instr_list) : instr_list(std::move(instr_list)) {}
    bh_ir(bh_instruction instr) {
        instr_list.push_back(std::move(instr));
    }
};

#endif
