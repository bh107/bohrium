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
#include "GenerateSourceCode.hpp"
#include "Accumulate.hpp"

bh_error Accumulate::bh_accumulate(bh_instruction* inst, UserFuncArg* userFuncArg)
{
    bh_view* out = &inst->operand[0];
    std::vector<bh_index> shape;
    for (int i = 0; i < out->ndim; ++i)
    {
        if(i == inst->constant.value.int64)
            continue;
        shape.push_back(out->shape[i]);
    }
    Kernel kernel = getKernel(inst, userFuncArg, shape);
    Kernel::Parameters kernelParameters;
    kernelParameters.push_back(std::make_pair(userFuncArg->operands[0], true));
    kernelParameters.push_back(std::make_pair(userFuncArg->operands[1], false));
    std::vector<size_t> globalShape;
    for (int i = shape.size()-1; i>=0; --i)
        globalShape.push_back(shape[i]);
    kernel.call(kernelParameters, globalShape);
    return BH_SUCCESS;
}

Kernel Accumulate::getKernel(bh_instruction* inst, 
                         UserFuncArg* userFuncArg,
                         std::vector<bh_index> shape)
{
#ifdef BH_TIMING
    bh_uint64 start = bh::Timer<>::stamp();
#endif
    std::string code = generateCode(inst, userFuncArg->operands[0]->type(), 
                                    userFuncArg->operands[1]->type(), shape);
#ifdef BH_TIMING
    userFuncArg->resourceManager->codeGen->add({start, bh::Timer<>::stamp()}); 
#endif
    size_t codeHash = string_hasher(code);
    KernelMap::iterator kit = kernelMap.find(codeHash);
    if (kit == kernelMap.end())
    {
        std::stringstream source, kname;
        kname << "accumulate" << std::hex << codeHash;
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
        Kernel kernel(userFuncArg->resourceManager, inst->operand[0].ndim-1, source.str(), kname.str());
        kernelMap.insert(std::make_pair(codeHash, kernel));
        return kernel;
    } else {
        return kit->second;
    }
}


std::string Accumulate::generateCode(bh_instruction* inst, 
                                     OCLtype outType, OCLtype inType,
                                     std::vector<bh_index> shape)
{
    bh_opcode opcode = 0;
    switch (inst->opcode)
    {
    case BH_ADD_ACCUMULATE:
        opcode = BH_ADD;
        break;
    case BH_MULTIPLY_ACCUMULATE:
        opcode = BH_MULTIPLY;
        break;
    default:
        assert(false);
    }
    bh_view* in = &inst->operand[1];
    std::stringstream source;
    std::vector<std::string> operands(3);
    operands[0] = "accu";
    operands[1] = "accu";
    operands[2] = "in[element]";
    source << "( __global " << oclTypeStr(outType) << "* out\n" 
        "                     , __global " << oclTypeStr(inType) << "* in)\n{\n";
    bh_view inn(inst->operand[1]);
    inn.ndim = in->ndim - 1;
    int i = 0;
    bh_int64 axis = inst->constant.value.int64;
    bh_int64 a = (axis)?0:1;
    while (a < in->ndim)
    {
        inn.shape[i] = in->shape[a];
        inn.stride[i++] = in->stride[a++];
        if (i == axis)
            ++a;
    }
    generateGIDSource(shape, source);
    source << "\tsize_t element = ";
    generateOffsetSource(inn, source);
    source << ";\n";
    source << "\t" << oclTypeStr(outType) << " accu = in[element];\n";
    source << "\tout[element] = accu;\n";
    source << "\tfor (int i = 1; i < " << in->shape[axis] << "; ++i)\n\t{\n";
    source << "\t\telement += " << in->stride[axis] << ";\n\t";
    generateInstructionSource(opcode, {outType, inType}, operands, source);
    source << "\t\tout[element] = accu;\n\t}\n}\n";
    return source.str();
}
