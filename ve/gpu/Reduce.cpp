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

#include <iostream>
#include <sstream>
#include <cassert>
#include <stdexcept>
#include <vector>
#include "GenerateSourceCode.hpp"
#include "Reduce.hpp"
#include "KernelParameter.hpp"
#include "Scalar.hpp"
#define MIN(a, b) ((a) < (b) ? (a) : (b))

typedef std::vector<std::pair<std::string, KernelParameter*>> ParameterList;
static ParameterList parameterList;
static bool accumulate;

bh_error Reduce::bh_reduce(const bh_instruction* inst, const UserFuncArg* userFuncArg)
{
    switch (inst->opcode)
    {
    case BH_ADD_ACCUMULATE:
    case BH_MULTIPLY_ACCUMULATE:
        accumulate = true;
        break;
    case BH_ADD_REDUCE:
    case BH_MULTIPLY_REDUCE:
    case BH_MINIMUM_REDUCE:
    case BH_MAXIMUM_REDUCE:
    case BH_LOGICAL_AND_REDUCE:
    case BH_BITWISE_AND_REDUCE:
    case BH_LOGICAL_OR_REDUCE:
    case BH_BITWISE_OR_REDUCE:
    case BH_LOGICAL_XOR_REDUCE:
    case BH_BITWISE_XOR_REDUCE:
        accumulate = false;
        break;
    default:
        assert(false);
    }
    const bh_view* out = &inst->operand[0];
    std::vector<bh_index> shape;
    for (int i = 0; i < out->ndim; ++i)
    {
        if(accumulate && i == inst->constant.value.int64)
            continue;
        shape.push_back(out->shape[i]);
    }

    const bh_view* in = &inst->operand[1];
    bh_view inn(*in);
    bh_int64 axis = inst->constant.value.int64;
    {
        inn.ndim = in->ndim - 1;
        int i = 0;
        bh_int64 a = (axis)?0:1;
        while (a < in->ndim)
        {
            inn.shape[i] = in->shape[a];
            inn.stride[i++] = in->stride[a++];
            if (i == axis)
                ++a;
        }
    }
#ifndef STATIC_KERNEL
    for (size_t i = 0; i < shape.size(); ++i)
    {
        {
            std::stringstream ss;
            ss << "ds" << shape.size()-(i+1);
            parameterList.push_back(std::make_pair(ss.str(), new Scalar(shape[i])));
        }{
            std::stringstream ss;
            ss << "v0s" << shape.size()-i;
            parameterList.push_back(std::make_pair(ss.str(), new Scalar(out->stride[i])));
        }{
            std::stringstream ss;
            ss << "v1s" << shape.size()-i;
            parameterList.push_back(std::make_pair(ss.str(), new Scalar(inn.stride[i])));
        }
    }
    parameterList.push_back(std::make_pair("v0s0", new Scalar(out->start)));
    parameterList.push_back(std::make_pair("v1s0", new Scalar(inn.start)));
    parameterList.push_back(std::make_pair("N", new Scalar(in->shape[axis])));
    parameterList.push_back(std::make_pair("S", new Scalar(in->stride[axis])));
#endif
    parameterList.push_back(std::make_pair("out",userFuncArg->operands[0]));
    parameterList.push_back(std::make_pair("in",userFuncArg->operands[1]));
    std::vector<bh_view> views;
    views.push_back(inst->operand[0]);
    views.push_back(inn);
    Kernel kernel = getKernel(inst, views, userFuncArg, shape);
    std::vector<size_t> globalShape;
    for (int i = shape.size()-1; i>=0 && shape.size()-i < 4; --i)
        globalShape.push_back(shape[i]);
    Kernel::Parameters kernelParameters;
    ParameterList::iterator pit = parameterList.begin();
    kernelParameters.push_back(std::make_pair(pit->second, true));
    for (++pit; pit != parameterList.end(); ++pit)
        kernelParameters.push_back(std::make_pair(pit->second, false));
    kernel.call(kernelParameters, globalShape);
    parameterList.clear();
    return BH_SUCCESS;
}

Kernel Reduce::getKernel(const bh_instruction* inst,
                         const std::vector<bh_view>& views,
                         const UserFuncArg* userFuncArg,
                         const std::vector<bh_index> shape)
{
#ifdef BH_TIMING
    bh_uint64 start = bh::Timer<>::stamp();
#endif
    std::string code = generateCode(inst, views, userFuncArg->operands[0]->type(), 
                                    userFuncArg->operands[1]->type(), shape);
#ifdef BH_TIMING
    userFuncArg->resourceManager->codeGen->add({start, bh::Timer<>::stamp()}); 
#endif
    size_t codeHash = string_hasher(code);
    KernelMap::iterator kit = kernelMap.find(codeHash);
    if (kit == kernelMap.end())
    {
        std::stringstream source, kname;
        kname << "reduce" << std::hex << codeHash;
        
        switch (userFuncArg->operands[0]->type())
        {
        case OCL_FLOAT16:
            source << "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n";
            break;
        case OCL_FLOAT64:
            source << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
            break;
        case OCL_COMPLEX64:
        case OCL_COMPLEX128:
            source << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
            source << "#include <ocl_complex.h>\n";
            break;
        default:
            break;
        }
        source << "__kernel void " << kname.str() << code;
        Kernel kernel(userFuncArg->resourceManager, MIN(shape.size(),3), source.str(), kname.str());
        kernelMap.insert(std::make_pair(codeHash, kernel));
        return kernel;
    } else {
        return kit->second;
    }
}


std::string Reduce::generateCode(const bh_instruction* inst, 
                                 const std::vector<bh_view>& views,
                                 const OCLtype outType, const OCLtype inType,
                                 const std::vector<bh_index> shape)
{
    bh_opcode opcode = 0;
    switch (inst->opcode)
    {
    case BH_ADD_REDUCE:
    case BH_ADD_ACCUMULATE:
        opcode = BH_ADD;
        break;
    case BH_MULTIPLY_REDUCE:
    case BH_MULTIPLY_ACCUMULATE:
        opcode = BH_MULTIPLY;
        break;
    case BH_MINIMUM_REDUCE:
        opcode = BH_MINIMUM;
        break;
    case BH_MAXIMUM_REDUCE:
        opcode = BH_MAXIMUM;
        break;
    case BH_LOGICAL_AND_REDUCE:
        opcode = BH_LOGICAL_AND;
        break;
    case BH_BITWISE_AND_REDUCE:
        opcode = BH_BITWISE_AND;
        break;
    case BH_LOGICAL_OR_REDUCE:
        opcode = BH_LOGICAL_OR;
        break;
    case BH_BITWISE_OR_REDUCE:
        opcode = BH_BITWISE_OR;
        break;
    case BH_LOGICAL_XOR_REDUCE:
        opcode = BH_LOGICAL_XOR;
        break;
    case BH_BITWISE_XOR_REDUCE:
        opcode = BH_BITWISE_XOR;
        break;
    default:
        assert(false);
    }
    std::stringstream source;
    std::vector<std::string> operands(3);
    operands[0] = "accu";
    operands[1] = "accu";
    operands[2] = "in[element]";
    source << "( ";
    // Add kernel parameters
    ParameterList::iterator pit = parameterList.begin();
    source << *(pit->second) << " " << pit->first;
    for (++pit; pit != parameterList.end(); ++pit)
    {
        source << "\n                     , " << *(pit->second) << " " << pit->first;
    }
    source << ")\n{\n";
    generateGIDSource(shape, source);
    std::stringstream indentss;
    indentss << "\t";
    for (size_t d = shape.size()-1; d > 2; --d)
    {
#ifdef STATIC_KERNEL
        source << indentss.str() << "for (int ids" << d << " = 0; ids" << d << " < " << 
            shape[shape.size()-(d+1)] << "; ++ids" << d << ")\n" << indentss.str() << "{\n";
#else
        source << indentss.str() << "for (int ids" << d << " = 0; ids" << d << " < ds" << d << 
            "; ++ids" << d << ")\n" << indentss.str() << "{\n";
#endif
        indentss << "\t";
    }
    std::string indent = indentss.str();
    source << indent << "size_t element = ";
    generateOffsetSource(views, 1, source);
    source << ";\n";
    source << indent << oclTypeStr(outType) << " accu = in[element];\n";
    if (accumulate)
        source << indent << "out[element] = accu;\n";
#ifdef STATIC_KERNEL
    const bh_view* in = &inst->operand[1];
    bh_int64 axis = inst->constant.value.int64;
    source << indent << "for (int i = 1; i < " << in->shape[axis] << "; ++i)\n" << indent 
           << "{\n" << indent << "\telement += " << in->stride[axis] << ";\n\t";
#else
    source << indent << "for (int i = 1; i < N; ++i)\n" << indent 
           << "{\n" << indent << "\telement += S;\n\t";
#endif
    generateInstructionSource(opcode, {outType, inType}, operands, indent, source);
    if (accumulate)
    {
        source << indent << "\tout[element] = accu;\n" << indent << "}\n";
    } else {
        source << indent << "}\n" << indent << "out[";
        generateOffsetSource(views, 0, source);
        source << "] = accu;\n";
    }
    source << indent.substr(1) << "}\n";
    for (size_t d = 3; d < shape.size(); ++d)
        source << indent.substr(d-1) << "}\n";
    return source.str();
}
