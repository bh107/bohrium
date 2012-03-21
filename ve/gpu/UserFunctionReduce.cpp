/*
 * Copyright 2012 Troels Blum <troels@blum.dk>
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
#include <sstream>
#include <cassert>
#include "GenerateSourceCode.hpp"
#include "UserFunctionReduce.hpp"


cphvb_error cphvb_reduce(cphvb_userfunc* arg, void* ve_arg)
{
    cphvb_reduce_type* reduceDef = (cphvb_reduce_type*)arg;
    UserFuncArg* userFuncArg = (UserFuncArg*)ve_arg;
    assert(reduceDef->nout = 1);
    assert(reduceDef->nin = 1);
    assert(reduceDef->operand[0]->ndim + 1 == reduceDef->operand[1]->ndim);
    assert(reduceDef->operand[0]->type == reduceDef->operand[1]->type);
    assert(userFuncArg->operandBase.size() == 2);
    UserFunctionReduce::run(reduceDef, userFuncArg);
    return CPHVB_SUCCESS;
}

void UserFunctionReduce::run(cphvb_reduce_type* reduceDef, UserFuncArg* userFuncArg)
{
    Kernel kernel = UserFunctionReduce::generateKernel(reduceDef, userFuncArg);
    Kernel::Parameters kernelParameters;
    kernelParameters.push_back(std::make_pair(userFuncArg->operandBase[0], true));
    kernelParameters.push_back(std::make_pair(userFuncArg->operandBase[1], false));
    std::vector<cphvb_index> shape = 
        std::vector<cphvb_index>(reduceDef->operand[0]->shape, 
                                 reduceDef->operand[0]->shape + reduceDef->operand[0]->ndim);
    kernel.call(kernelParameters, shape);
}

Kernel UserFunctionReduce::generateKernel(cphvb_reduce_type* reduceDef, UserFuncArg* userFuncArg)
{
    std::stringstream ss;
    ss << "reduce" << kernel++;
    std::string code = UserFunctionReduce::generateCode(reduceDef, userFuncArg->operandBase, ss.str());
    std::vector<OCLtype> signature;
    signature.push_back(OCL_BUFFER);
    signature.push_back(OCL_BUFFER);
    return Kernel(userFuncArg->resourceManager, reduceDef->operand[0]->ndim , signature, code, ss.str());
}


std::string UserFunctionReduce::generateCode(cphvb_reduce_type* reduceDef, 
                                             const std::vector<BaseArray*>& operandBase,
                                             const std::string& kernelName)
{
    cphvb_array* out = reduceDef->operand[0];
    cphvb_array* in = reduceDef->operand[1];
    std::stringstream source;
    std::vector<std::string> operands(3);
    operands[0] = "accu";
    operands[1] = "accu";
    operands[2] = "in[element]";
    source << "__kernel void " << kernelName << "(" <<
        " __global " << oclTypeStr(operandBase[0]->type()) << "* out\n" 
        "                     , __global " << oclTypeStr(operandBase[1]->type()) << "* in"
        //", __local  " << oclTypeStr(oclType(CPHVB_INDEX)) << " axis" 
        ")\n{\n";
    
    source << "\tsize_t element = ";
    int i = 0;
    int a = (reduceDef->axis)?0:1;
    source << "get_global_id(" << i++ << ")*" << in->stride[a++];
    if (i == reduceDef->axis)
        ++a;
    while (a < in->ndim)
    {
        source << " + get_global_id(" << i++ << ")*" << in->stride[a++];
        if (i == reduceDef->axis)
            ++a;
    }
    source << " + " << in->start << ";\n";
    source << "\t" << oclTypeStr(operandBase[0]->type()) << " accu = in[element];\n";
    source << "\tfor (int i = 1; i < " << in->shape[reduceDef->axis] << "; ++i)\n\t{\n";
    source << "\t\telement += " << in->stride[reduceDef->axis] << ";\n\t";
    generateInstructionSource(reduceDef->opcode, operandBase[0]->type(), operands, source);
    source << "\t}\n\tout[";
    i = 0;
    source << "get_global_id(" << i << ")*" << out->stride[i];
    for (++i; i < out->ndim; ++i)
    {
        source << " + get_global_id(" << i << ")*" << out->stride[i];
    }
    source << " + " << out->start << "] = accu;\n}\n";
    return source.str();
}
