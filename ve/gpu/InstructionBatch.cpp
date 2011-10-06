/*
 * Copyright 2011 Troels Blum <troels@blum.dk>
 *
 * This file is part of cphVB <http://code.google.com/p/cphvb/>.
 *
 * cphVB is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * cphVB is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with cphVB. If not, see <http://www.gnu.org/licenses/>.
 */

#include <iostream>
#include <cassert>
#include <cphvb.h>
#include "InstructionBatch.hpp"

InstructionBatch::InstructionBatch(cphvb_instruction* inst, const std::vector<BaseArray*>& operandBase)
{
    accept(inst, operandBase);
}

void InstructionBatch::accept(cphvb_instruction* inst, const std::vector<BaseArray*>& operandBase)
{
    int nops = cphvb_operands(inst->opcode);
    assert(nops > 0);
    output[operandBase[0]] = inst->operand[0];
    for (int i = 1; i < nops; ++i)
    {
        cphvb_array* operand = inst->operand[i];
        if (operand != CPHVB_CONSTANT)
        {
            input.insert(operandBase[i]);
        }
    }
    instructions.push_back(inst);
}

void InstructionBatch::add(cphvb_instruction* inst, const std::vector<BaseArray*>& operandBase)
{
    int nops = cphvb_operands(inst->opcode);
    assert(nops > 0);
    for (int i = 1; i < nops; ++i)
    {
        cphvb_array* operand = inst->operand[i];
        if (operand != CPHVB_CONSTANT)
        {
            Output::iterator it = output.find(operandBase[i]);
            if (it != output.end() && it->second != operand)
            {
                throw BatchException(0);
            }
        }
    }
    accept(inst, operandBase);
}

bool InstructionBatch::use(BaseArray* baseArray)
{
    if (output.find(baseArray) == output.end() && input.find(baseArray) == input.end())
    { 
        return true;
    }
    return false;
}

