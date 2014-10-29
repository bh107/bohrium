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
#include "Scalar.hpp"
#include "StringHasher.hpp"
#define MIN(a, b) ((a) < (b) ? (a) : (b))

SourceKernelCall Reduce::generateKernel(const bh_instruction* inst, 
                                        const std::vector<KernelParameter*> operands)
{
#ifdef BH_TIMING
    bh_uint64 start = bh::Timer<>::stamp();
#endif
    bool accumulate;
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

    std::vector<KernelParameter*> sizeParameters;
    std::stringstream defines;
    std::stringstream functionDeclaration;

    functionDeclaration << "(\n\t  " << *operands[0] << " out\n\t, " << *operands[1] << 
        " in\n#ifndef FIXED_SIZE";
    Kernel::Parameters valueParameters;
    valueParameters.push_back(std::make_pair(operands[0], true));
    valueParameters.push_back(std::make_pair(operands[1], false));

    for (size_t i = 0; i < shape.size(); ++i)
    {
        {
            std::stringstream ss;
            ss << "ds" << shape.size()-(i+1);
            Scalar* s = new Scalar(shape[i]);
            (defines << "#define " << ss.str() << " " <<= *s) << "\n";
            sizeParameters.push_back(s);
            functionDeclaration << "\n\t, "  << *s << " " << ss.str();
        }{
            std::stringstream ss;
            ss << "v0s" << shape.size()-i;
            Scalar* s = new Scalar(out->stride[i]);
            (defines << "#define " << ss.str() << " " <<= *s) << "\n";
            sizeParameters.push_back(s);
            functionDeclaration << "\n\t, " << *s << " " << ss.str();
        }{
            std::stringstream ss;
            ss << "v1s" << shape.size()-i;
            Scalar* s = new Scalar(inn.stride[i]);
            (defines << "#define " << ss.str() << " " <<= *s) << "\n";
            sizeParameters.push_back(s);
            functionDeclaration << "\n\t, " << *s << " " << ss.str();
        }
    }
    Scalar* s = new Scalar(out->start);
    (defines << "#define v0s0 " <<= *s) << "\n";
    sizeParameters.push_back(s);
    functionDeclaration << "\n\t, " << *s << " v0s0";
    s = new Scalar(inn.start);
    (defines << "#define v1s0 " <<= *s) << "\n";
    sizeParameters.push_back(s);
    functionDeclaration << "\n\t, " << *s << " v1s0";
    s = new Scalar(in->shape[axis]);
    (defines << "#define N " <<= *s) << "\n";
    sizeParameters.push_back(s);
    functionDeclaration << "\n\t, " << *s << " N";
    s = new Scalar(in->stride[axis]);
    (defines << "#define S " <<= *s) << "\n";
    sizeParameters.push_back(s);
    functionDeclaration << "\n\t, " << *s << " S";
    functionDeclaration << "\n#endif\n)\n";

    std::string functionBody = generateFunctionBody(inst, operands[0]->type(), operands[1]->type(), 
                                                    shape, accumulate);
    size_t functionID = string_hasher(functionBody);
    size_t literalID = string_hasher(defines.str());

    std::vector<size_t> kernelShape;
    for (int i = shape.size()-1; i>=0 && shape.size()-i < 4; --i)
        kernelShape.push_back(shape[i]);
    
    std::stringstream source;
    switch (operands[0]->type())
    {
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
    source << "#ifdef FIXED_SIZE\n" << defines.str() << "#endif\n" << 
        "__kernel void\n#ifndef FIXED_SIZE\nkernel" << std::hex << functionID <<
        "\n#else\nkernel" << std::hex << functionID << "_\n#endif\n" << functionDeclaration.str() << 
        "\n" << functionBody;
    
#ifdef BH_TIMING
    resourceManager->codeGen->add({start, bh::Timer<>::stamp()}); 
#endif
    return SourceKernelCall(KernelID(functionID, literalID), kernelShape, source.str(), 
                            sizeParameters, valueParameters);
}

std::string Reduce::generateFunctionBody(const bh_instruction* inst, 
                                         const OCLtype outType, const OCLtype inType,
                                         const std::vector<bh_index>& shape,
                                         bool accumulate)
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
    source << "{\n";
    generateGIDSource(shape.size(), source);
    std::stringstream indentss;
    indentss << "\t";
    for (size_t d = shape.size()-1; d > 2; --d)
    {
        source << indentss.str() << "for (int ids" << d << " = 0; ids" << d << " < ds" << d << 
            "; ++ids" << d << ")\n" << indentss.str() << "{\n";
        indentss << "\t";
    }
    std::string indent = indentss.str();
    source << indent << "size_t element = ";
    generateOffsetSource(shape.size(), 1, source);
    source << ";\n";
    source << indent << oclTypeStr(outType) << " accu = in[element];\n";
    if (accumulate)
        source << indent << "out[element] = accu;\n";
    source << indent << "for (int i = 1; i < N; ++i)\n" << indent 
           << "{\n" << indent << "\telement += S;\n\t";
    generateInstructionSource(opcode, {outType, inType}, operands, indent, source);
    if (accumulate)
    {
        source << indent << "\tout[element] = accu;\n" << indent << "}\n";
    } else {
        source << indent << "}\n" << indent << "out[";
        generateOffsetSource(shape.size(), 0, source);
        source << "] = accu;\n";
    }
    source << indent.substr(1) << "}\n";
    for (size_t d = 3; d < shape.size(); ++d)
        source << indent.substr(d-1) << "}\n";
    return source.str();
}
