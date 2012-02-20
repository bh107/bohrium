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

void InstructionBatch::add(cphvb_instruction* inst, const std::vector<BaseArray*>& operandBase)
{
    assert(inst->operand[0]->ndim > 0);

    // Check that the shape matches
    if (!shapeMatch(inst->operands[0]->ndim, inst->operands[0]->shape))
        throw BatchException(0);

    // If any operand's base is already used as output, it has to be alligned.
    for (int op = 0; op < operandBase.size(); ++op)
    {
        if (inst->operand[op]->ndim != 0)
        {
            if (output.find(operandBase[op]) != output.end())
            {
                if (!sameView(oit.second(), inst->operand[op]))
                    throw BatchException(0);
            }
        }
    }

    // If the output operans is allready used as input it has to be alligned 
    std::pair<inputMap::iterator, inputMap::iterator> irange = input.equal_range(operandBase[0]);
    InputMap::Iterator iit = irange.first;
    for (; iit != irange.second; ++iit)
    {
        if (!sameView(iit.second(), inst->operand[0]))
            throw BatchException(0);
    }

    
    // OK so we can accept the instruction
    output[operandBase[0]] = inst->operand[0];
    
    // HERE: also inser into input and stuff

    for (int op = 0; op < operandBase.size(); ++op)
    {
        cphvb_array* base = cphvb_base_array(inst->operand[op]);
        if (inst->operand[op]->ndim != 0 && kernelParameters.find(base) == kernelParameters.end()))
        {
            std::stringstream ss;
            ss << "a" << arraynum++;
            kernelParameters[base] = std::make_pair(operandBase[op],ss.str()); 
        }
    }    
}

std::string InstructionBatch::generateCode()
{
    std::stringstream source;
    source << "__kernel void kernel" << kernel++ << "(";
    ParamMap::iterator kpit = kernelParameters.start();
    source << "__global " << oclTypeStr(kpit.second().first->bufferType) << "* " << kpit.second().second;
    for (++kpit; kpit != kernelParameters.end(); ++kpit)
    {
        source << "\n                       __global " << 
            oclTypeStr(kpit.second().first->bufferType) << "* " << kpit.second().second; 
    }
    source << ")\n{\n";
    
    for (std::vector<cphvb_instruction*>::iterator iit = instructions.start(), iit != instructions.end(); ++iit)
    {
        generateInstructionSource(*iit, source);
    }
    source << "}\n";
}

void InstructionBatch::generateInstructionSource(cphvb_instruction* inst, std::ostream& source)
{
    switch(inst->opcode)
    {
    case CPHVB_ADD:
        source << kernelParameters[inst->operand[0]]->second << " = " <<
            kernelParameters[inst->operand[1]]->second << " + " <<
            kernelParameters[inst->operand[2]]->second << ";\n";
        break;
    default:
        throw std::runtime_error("Instruction not supported.");
    }
}

void InstructionBatch::generateOffsetSource(cphvb_array* operand, std::ostream& source)
{
    if (operand->ndim > 2)
    {
        source << "get_global_id(2)*" << operand->stride[2] << " + ";
    }
    if (operand->ndim > 1)
    {
        source << "get_global_id(1)*" << operand->stride[1] << " + ";
    }
    source << "get_global_id(0)*" << operand->stride[0] << " + " << operand->start;
    
    
}
