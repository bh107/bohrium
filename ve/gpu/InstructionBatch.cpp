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

bool InstructionBatch::shapeMatch(cphvb_intp ndim,const cphvb_index dims[])
{
    if (ndim == shape.size())
    {
        for (int i = 0; i < ndim; ++i)
        {
            if (dims[i] != shape[i])
                return false;
        }
        return true;
    }
    return false;
}

bool InstructionBatch::sameView(const cphvb_array* a, const cphvb_array* b)
{
    //assumes the the views shape are the same and they have the same base
    if (a == b)
        return true;
    if (a->start != b->start)
        return false;
    for (int i = 0; i < shape.size(); ++i)
    {
        if (a->stride[i] != b->stride[i])
            return false;
    }
    return true;
}

InstructionBatch::InstructionBatch(cphvb_instruction* inst, const std::vector<BaseArray*>& operandBase)
{
    shape = std::vector<cphvb_index>(inst->operand[0]->shape, inst->operand[0]->shape + inst->operand[0]->ndim);
    accept(inst, operandBase);
}

bool InstructionBatch::accept(cphvb_instruction* inst, const std::vector<BaseArray*>& operandBase)
{
    if (!shapeMatch(inst->operands[0]->ndim, inst->operands[0]->shape))
        return false;
    std::map<BaseArray*, cphvb_array*>::iterator oit;
    for (int op = 0; op < operandBase.size(); op++)
    {
        if (inst->operand[op]->ndim != 0)
        {
            oit = output.find(operandBase[op]);
            if (oit != output.end())
            {
                if (!sameView(oit.second(), inst->operand[op]))
                    return false;
            }
        }
    }
    return true;
}

void InstructionBatch::add(cphvb_instruction* inst, const std::vector<BaseArray*>& operandBase)
{
    // Check that the shape matches
    if (!shapeMatch(inst->operands[0]->ndim, inst->operands[0]->shape))
        throw BatchException(0);

    // If any operand's base is already used as output, it has to be alligned.
    std::map<BaseArray*, cphvb_array*>::iterator oit;
    for (int op = 0; op < operandBase.size(); op++)
    {
        if (inst->operand[op]->ndim != 0)
        {
            oit = output.find(operandBase[op]);
            if (oit != output.end())
            {
                if (!sameView(oit.second(), inst->operand[op]))
                    throw BatchException(0);
            }
        }
    }
    
    // 
}


std::string InstructionBatch::generateCode()
{
    std::ostream os;
    os << "__kernel void kernel" << kernel++ << "(\n";
    
}
