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
    assert(reduceDef->operand[0]->ndim + 1 == reduceDef->operand[1]->ndim || cphvb_scalar(reduceDef->operand[0]));
    assert(reduceDef->operand[0]->type == reduceDef->operand[1]->type);
    assert(userFuncArg->operandBase.size() == 2);
    UserFunctionReduce::run(reduceDef, userFuncArg);
    return CPHVB_SUCCESS;
}

#define TPB 128
#define BPG 128

void UserFunctionReduce::run(cphvb_reduce_type* reduceDef, UserFuncArg* userFuncArg)
{
    cphvb_array* out = reduceDef->operand[0];
    std::vector<cphvb_index> shape;
    if (cphvb_scalar(out))
        shape.push_back(BPG*TPB);
    else
        shape = std::vector<cphvb_index>(out->shape, out->shape + out->ndim);
    Kernel kernel = generateKernel(reduceDef, userFuncArg, shape);
    Kernel::Parameters kernelParameters;
    BaseArray* outArray;
    if (cphvb_scalar(out))
    {
        cphvb_array out_temp;
        out_temp.base = NULL;
        out_temp.type = out->type;
        out_temp.ndim = 1;
        out_temp.start = 0;
        out_temp.shape[0] = BPG*TPB;
        out_temp.stride[0] = 1;
        out_temp.data = NULL;
        outArray = new BaseArray(&out_temp, userFuncArg->resourceManager);
    }
    else 
        outArray = userFuncArg->operandBase[0];
    kernelParameters.push_back(std::make_pair(outArray, true));
    kernelParameters.push_back(std::make_pair(userFuncArg->operandBase[1], false));
    if (cphvb_scalar(out))
    {
        std::vector<size_t> localShape;
        localShape.push_back(TPB);
        kernel.call(kernelParameters, shape, localShape);
        reduce1Dfinnish(reduceDef->opcode, outArray, userFuncArg->operandBase[0]);
    } else
        kernel.call(kernelParameters, shape);
}

Kernel UserFunctionReduce::generateKernel(cphvb_reduce_type* reduceDef, 
                                          UserFuncArg* userFuncArg,
                                          const std::vector<cphvb_index>& shape)
{
#ifdef STATS
    timeval start, end;
    gettimeofday(&start,NULL);
#endif
    std::string code = generateCode(reduceDef, userFuncArg->operandBase, shape);
#ifdef STATS
    gettimeofday(&end,NULL);
    userFuncArg->resourceManager->batchSource += 
        (end.tv_sec - start.tv_sec)*1000000.0 + (end.tv_usec - start.tv_usec);
#endif
    KernelMap::iterator kit = kernelMap.find(code);
    if (kit == kernelMap.end())
    {
        std::stringstream source, kname;
        kname << "reduce" << kernelNo++;
        source << "__kernel void " << kname.str() << code;
        Kernel kernel(userFuncArg->resourceManager, reduceDef->operand[0]->ndim, source.str(), kname.str());
        kernelMap.insert(std::make_pair(code, kernel));
        return kernel;
    } else {
        return kit->second;
    }
}


std::string UserFunctionReduce::generateCode(cphvb_reduce_type* reduceDef, 
                                             const std::vector<BaseArray*>& operandBase,
                                             const std::vector<cphvb_index>& shape)
{
    cphvb_array* out = reduceDef->operand[0];
    cphvb_array* in = reduceDef->operand[1];
    std::stringstream source;
    std::vector<std::string> operands(3);
    operands[0] = "accu";
    operands[1] = "accu";
    operands[2] = "in[element]";
    source << "( __global " << oclTypeStr(operandBase[0]->type()) << "* out\n" 
        "                     , __global " << oclTypeStr(operandBase[1]->type()) << "* in)\n{\n";
    if (cphvb_scalar(out))
    {
        source << "\tconst size_t gidx = get_global_id(0);\n";
        source << "\tsize_t element = gidx*" << in->stride[reduceDef->axis] << " + " << in->start << ";\n";
        source << "\t" << oclTypeStr(operandBase[0]->type()) << " accu = 0;\n";
        source << "\tfor (int i = 0; i < " << in->shape[reduceDef->axis]/shape[0] << "; ++i)\n\t{\n\t";
        generateInstructionSource(reduceDef->opcode, operandBase[0]->type(), operands, source);
        source << "\t\telement += " << in->stride[reduceDef->axis] * BPG*TPB << ";\n\t}\n";
        source << "\tif (element < " << in->start +  in->shape[reduceDef->axis] * in->stride[reduceDef->axis] 
               << ")\n\t{\n\t";
        generateInstructionSource(reduceDef->opcode, operandBase[0]->type(), operands, source);
    } else {
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
        source << "\t" << oclTypeStr(operandBase[0]->type()) << " accu = in[element];\n";
        source << "\tfor (int i = 1; i < " << in->shape[reduceDef->axis] << "; ++i)\n\t{\n";
        source << "\t\telement += " << in->stride[reduceDef->axis] << ";\n\t";
        generateInstructionSource(reduceDef->opcode, operandBase[0]->type(), operands, source);
    }
    source << "\t}\n\tout[";
    generateOffsetSource(out, source);
    source << "] = accu;\n}\n";
    return source.str();
}

void UserFunctionReduce::reduce1Dfinnish(cphvb_opcode opcode, BaseArray* tmpArray, BaseArray* outArray)
{
    if (opcode != CPHVB_ADD)
        throw std::runtime_error("Opcode not supported for 1D reduce");
    tmpArray->sync();
    cphvb_array* temp = tmpArray->getSpec();
    cphvb_array* out = outArray->getSpec();
    cphvb_data_malloc(out);
    switch(temp->type)
    {
    case CPHVB_INT32:
    {
        cphvb_int32* in = (cphvb_int32*)temp->data;
        cphvb_int32 accu = in[0];
        for (int i = 1; i < BPG*TPB; ++i)
            accu += in[i];
        *(cphvb_int32*)out->data = accu;
        break;
    }
    case CPHVB_INT64:
    {
        cphvb_int64* in = (cphvb_int64*)temp->data;
        cphvb_int64 accu = in[0];
        for (int i = 1; i < BPG*TPB; ++i)
            accu += in[i];
        *(cphvb_int64*)out->data = accu;
        break;
    }
    case CPHVB_FLOAT32:
    {
        cphvb_float32* in = (cphvb_float32*)temp->data;
        cphvb_float32 accu = in[0];
        for (int i = 1; i < BPG*TPB; ++i)
            accu += in[i];
        *(cphvb_float32*)out->data = accu;
        break;
    }
    case CPHVB_FLOAT64:
    {
        cphvb_float64* in = (cphvb_float64*)temp->data;
        cphvb_float64 accu = in[0];
        for (int i = 1; i < BPG*TPB; ++i)
            accu += in[i];
        *(cphvb_float64*)out->data = accu;
        break;
    }
    default:
        throw std::runtime_error("Data type not supported for 1D reduce");
    }
}

