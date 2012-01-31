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

bool InstructionBatch::sameShape(cphvb_intp ndim,const cphvb_index dims[])
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

bool InstructionBatch::accept(cphvb_instruction* inst, const std::vector<BaseArray*>& operandBase)
{
    if (!sameShape(inst->operands[0]->ndim, inst->operands[0]->shape))
        return false;
    std::map<BaseArray*, cphvb_array*>::iterator oit;
    std::multimap<BaseArray*, cphvb_array*>::iterator iit;
    std::pair<std::multimap<BaseArray*, cphvb_array*>::iterator, 
              std::multimap<BaseArray*, cphvb_array*>::iterator> irange;
    for (int op = 0; op < operandBase.size(); op++)
    {
        oit = output.find(operandBase[op]);
        if (oit != output.end())
        {
            if ()
        }

        irange = input.equal_range(operandBase[op]);
        for (iit = irange.first; iit != irange.second; ++iit)
        {
            
        }
    }
}

InstructionBatch::InstructionBatch(cphvb_instruction* inst, const std::vector<BaseArray*>& operandBase)
{
    shape = std::vector<cphvb_index>(inst->operand[0]->shape, inst->operand[0]->shape + inst->operand[0]->ndim);
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

