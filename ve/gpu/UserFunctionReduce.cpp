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
#include <stdexcept>
#include "GenerateSourceCode.hpp"
#include "UserFunctionReduce.hpp"

cphvb_error cphvb_reduce(cphvb_userfunc* arg, void* ve_arg)
{
    cphvb_reduce_type* reduceDef = (cphvb_reduce_type*)arg;
    UserFuncArg* userFuncArg = (UserFuncArg*)ve_arg;
    assert(reduceDef->nout = 1);
    assert(reduceDef->nin = 1);
    assert(reduceDef->operand[0]->ndim + 1 == reduceDef->operand[1]->ndim || cphvb_is_scalar(reduceDef->operand[0]));
    assert(userFuncArg->operands.size() == 2);
    if (cphvb_is_scalar(reduceDef->operand[0]))
    {
        static_cast<BaseArray*>(userFuncArg->operands[1])->sync();
        cphvb_error err = cphvb_compute_reduce(arg,NULL);
        if (err == CPHVB_SUCCESS)
            static_cast<BaseArray*>(userFuncArg->operands[0])->update();
        return err;
    }
    else
    {
        UserFunctionReduce::reduce(reduceDef, userFuncArg);
        return CPHVB_SUCCESS;
    }
}

void UserFunctionReduce::reduce(cphvb_reduce_type* reduceDef, UserFuncArg* userFuncArg)
{
    cphvb_array* out = reduceDef->operand[0];
    std::vector<cphvb_index> shape = std::vector<cphvb_index>(out->shape, out->shape + out->ndim);
    Kernel kernel = getKernel(reduceDef, userFuncArg, shape);
    Kernel::Parameters kernelParameters;
    kernelParameters.push_back(std::make_pair(userFuncArg->operands[0], true));
    kernelParameters.push_back(std::make_pair(userFuncArg->operands[1], false));
    std::vector<size_t> globalShape;
    for (int i = shape.size()-1; i>=0; --i)
        globalShape.push_back(shape[i]);
    kernel.call(kernelParameters, globalShape);
}

Kernel UserFunctionReduce::getKernel(cphvb_reduce_type* reduceDef, 
                                     UserFuncArg* userFuncArg,
                                     std::vector<cphvb_index> shape)
{
#ifdef STATS
    timeval start, end;
    gettimeofday(&start,NULL);
#endif
    std::string code = generateCode(reduceDef, userFuncArg->operands[0]->type(), 
                                    userFuncArg->operands[1]->type(), shape);
#ifdef STATS
    gettimeofday(&end,NULL);
    userFuncArg->resourceManager->batchSource += 
        (end.tv_sec - start.tv_sec)*1000000.0 + (end.tv_usec - start.tv_usec);
#endif
    size_t codeHash = string_hasher(code);
    KernelMap::iterator kit = kernelMap.find(codeHash);
    if (kit == kernelMap.end())
    {
        std::stringstream source, kname;
        kname << "reduce" << std::hex << codeHash;
        if (userFuncArg->operands[0]->type() == OCL_FLOAT16 || 
            userFuncArg->operands[1]->type() == OCL_FLOAT16)
        {
            source << "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n";
        }
        else if (userFuncArg->operands[0]->type() == OCL_FLOAT64 ||
                 userFuncArg->operands[0]->type() == OCL_FLOAT64)
        {
            source << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
        }
        source << "__kernel void " << kname.str() << code;
        Kernel kernel(userFuncArg->resourceManager, reduceDef->operand[0]->ndim, source.str(), kname.str());
        kernelMap.insert(std::make_pair(codeHash, kernel));
        return kernel;
    } else {
        return kit->second;
    }
}


std::string UserFunctionReduce::generateCode(cphvb_reduce_type* reduceDef, 
                                             OCLtype outType, OCLtype inType,
                                             std::vector<cphvb_index> shape)
{
    cphvb_array* out = reduceDef->operand[0];
    cphvb_array* in = reduceDef->operand[1];
    std::stringstream source;
    std::vector<std::string> operands(3);
    operands[0] = "accu";
    operands[1] = "accu";
    operands[2] = "in[element]";
    source << "( __global " << oclTypeStr(outType) << "* out\n" 
        "                     , __global " << oclTypeStr(inType) << "* in)\n{\n";
    cphvb_array inn(*in);
    inn.ndim = in->ndim - 1;
    int i = 0;
    int a = (reduceDef->axis)?0:1;
    while (a < in->ndim)
    {
        inn.shape[i] = in->shape[a];
        inn.stride[i++] = in->stride[a++];
        if (i == reduceDef->axis)
            ++a;
    }
    generateGIDSource(shape, source);
    source << "\tsize_t element = ";
    generateOffsetSource(&inn, source);
    source << ";\n";
    source << "\t" << oclTypeStr(outType) << " accu = in[element];\n";
    source << "\tfor (int i = 1; i < " << in->shape[reduceDef->axis] << "; ++i)\n\t{\n";
    source << "\t\telement += " << in->stride[reduceDef->axis] << ";\n\t";
    generateInstructionSource(reduceDef->opcode, outType, operands, source);
    source << "\t}\n\tout[";
    generateOffsetSource(out, source);
    source << "] = accu;\n}\n";
    return source.str();
}
