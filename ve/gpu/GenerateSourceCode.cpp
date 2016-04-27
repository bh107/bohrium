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

#include <cassert>
#include <stdexcept>
#include <algorithm>

#include "GenerateSourceCode.hpp"
#include "bh_ve_gpu.h"

void generateGIDSource(size_t kdims, std::ostream& source)
{
    assert(kdims > 0);
    assert(kdims <= 3);
    for (size_t d = 1; d <= kdims; ++d)
    {
        source << "\tconst size_t idd" << d << " = get_global_id(" << d-1 << ");\n";
        source << "\tif (idd" << d << " >= ds" << d << ")\n\t\treturn;\n";
    }
}

void generateOffsetSource(size_t cdims, bh_index vdims, size_t id, std::ostream& source)
{
    assert(vdims > 0);
    assert(cdims > 0);
    source << "(";
    for (bh_index d = cdims; d > vdims; --d)
    {
        source << "idd" << d << "*ds" << d-1 << "+";
    }
    bh_index dims = std::min((bh_index)cdims,vdims);
    source << "idd" << dims << ")*v" << id << "s" << dims << " + ";
    for (bh_index d = dims-1; d > 0; --d)
    {
        source << "idd" << d << "*v" << id << "s" << d << " + ";
    }
    source << "v" << id << "s0";
}

void generateIndexSource(size_t cdims, bh_index vdims, size_t id, std::ostream& source)
{
    source << "size_t v" << id << "idx = ";
    generateOffsetSource(cdims, vdims, id, source);
    source << ";\n";
}

void generateSaveSource(size_t aid, size_t vid, std::ostream& source)
{
    source << "a" << aid << "[v" << vid << "idx] = v" << vid << ";\n";
}

void generateLoadSource(size_t aid, size_t vid, OCLtype type, std::ostream& source)
{
    source << oclTypeStr(type) << " v" << vid <<  " = a" << aid << "[v" << vid << "idx];\n";
}

void generateElementNumber(const std::vector<size_t>& dimOrder, std::ostream& source)
{
    size_t cdims = dimOrder.size();
    assert(cdims > 0);
    for (size_t d = cdims; d > 0; --d)
    {
        source << "idd" << d;
        if (dimOrder[cdims-d])
            source << "*ds" << dimOrder[cdims-d];
        if (d > 1)
            source << " + ";
    }
}

void generateNeutral(bh_opcode opcode,OCLtype type, std::ostream& source)
{
    switch (type)
    {
    case OCL_COMPLEX64:
        source << "(float2)(0.0f, ";
        break;
    case OCL_COMPLEX128:
        source << "(double2)(0.0, ";
        break;
    default:
        break;
    }
    switch (opcode)
    {
    case BH_ADD_ACCUMULATE:
    case BH_ADD_REDUCE:
    case BH_LOGICAL_OR_REDUCE:
    case BH_BITWISE_OR_REDUCE:
    case BH_LOGICAL_XOR_REDUCE:
    case BH_BITWISE_XOR_REDUCE:
        source << "0";
        break;
    case BH_MULTIPLY_ACCUMULATE:
    case BH_MULTIPLY_REDUCE:
    case BH_LOGICAL_AND_REDUCE:
        source << "1";
        break;
    case BH_BITWISE_AND_REDUCE:
        source << "~0";
        break;
    case BH_MINIMUM_REDUCE:
        switch (type)
        {
        case OCL_INT8:
            source << "CHAR_MAX";
            break;
        case OCL_INT16:
            source << "SHRT_MAX";
                break;
        case OCL_INT32:
            source << "INT_MAX";
                break;
        case OCL_INT64:
            source << "LONG_MAX";
                break;
        case OCL_UINT8:
            source << "UCHAR_MAX";
            break;
        case OCL_UINT16:
            source << "USHRT_MAX";
            break;
        case OCL_UINT32:
            source << "UINT_MAX";
            break;
        case OCL_UINT64:
            source << "ULONG_MAX";
            break;
        case OCL_FLOAT32:
            source << "FLT_MAX";
            break;
        case OCL_FLOAT64:
            source << "DBL_MAX";
            break;
        default:
            assert(false);
        }
        return;
    case BH_MAXIMUM_REDUCE:
        switch (type)
        {
        case OCL_INT8:
            source << "CHAR_MIN";
            break;
        case OCL_INT16:
            source << "SHRT_MIN";
            break;
        case OCL_INT32:
            source << "INT_MIN";
            break;
        case OCL_INT64:
            source << "LONG_MIN";
            break;
        case OCL_UINT8:
            source << "0";
            break;
        case OCL_UINT16:
            source << "0";
            break;
        case OCL_UINT32:
            source << "0";
            break;
        case OCL_UINT64:
            source << "0";
            break;
        case OCL_FLOAT32:
            source << "FLT_MIN";
            break;
        case OCL_FLOAT64:
            source << "DBL_MIN";
            break;
        default:
            assert(false);
        }
        return;
    default:
        assert(false);
    }
    switch (type)
    {
    case OCL_INT64:
        source << "l";
        break;
    case OCL_UINT8:
    case OCL_UINT16:
    case OCL_UINT32:
        source << "u";
        break;
    case OCL_UINT64:
        source << "ul";
        break;
    case OCL_FLOAT32:
        source << ".0f";
        break;
    case OCL_FLOAT64:
        source << ".0";
        break;
    case OCL_COMPLEX64:
        source << ".0f)";
        break;
    case OCL_COMPLEX128:
        source << ".0)";
        break;
    default:
        break;
    }
}

#define TYPE ((type[1] == OCL_COMPLEX64) ? "float" : "double")
void generateInstructionSource(const bh_opcode opcode,
                               const std::vector<OCLtype>& type,
                               const std::vector<std::string>& parameters,
                               const std::string& indent,
                               std::ostream& source)
{
    if (isComplex(type[0]) || (type.size()>1 && isComplex(type[1])))
    {

        switch(opcode)
        {
        case BH_ADD:
        case BH_ADD_REDUCE:
        case BH_ADD_ACCUMULATE:
            source << indent << "CADD(" << parameters[0] << ", " << parameters[1] << ", " <<
                parameters[2] << ")\n";
            break;
        case BH_SUBTRACT:
            source << indent << "CSUB(" << parameters[0] << ", " << parameters[1] << ", " <<
                parameters[2] << ")\n";
            break;
        case BH_MULTIPLY:
        case BH_MULTIPLY_REDUCE:
        case BH_MULTIPLY_ACCUMULATE:
            source << indent << "CMUL(" << parameters[0] << ", " << parameters[1] << ", " <<
                parameters[2] << ")\n";
            break;
        case BH_DIVIDE:
            source << indent << "CDIV(" << TYPE << ", " << parameters[0] << ", " << parameters[1] << ", " <<
                parameters[2] << ")\n";
            break;
        case BH_POWER:
            source << indent << "CPOW(" << TYPE << ", " << parameters[0] << ", " << parameters[1] << ", " <<
                parameters[2] << ")\n";
            break;
        case BH_EQUAL:
            source << indent << "CEQ(" << parameters[0] << ", " << parameters[1] << ", " <<
                parameters[2] << ")\n";
            break;
        case BH_NOT_EQUAL:
            source << indent << "CNEQ(" << parameters[0] << ", " << parameters[1] << ", " <<
                parameters[2] << ")\n";
            break;
        case BH_COS:
            source << indent << "CCOS(" << TYPE << ", " << parameters[0] << ", " << parameters[1] << ")\n";
            break;
        case BH_SIN:
            source << indent << "CSIN(" << TYPE << ", " << parameters[0] << ", " << parameters[1] << ")\n";
            break;
        case BH_TAN:
            source << indent << "CTAN(" << TYPE << ", " << parameters[0] << ", " << parameters[1] << ")\n";
            break;
        case BH_COSH:
            source << indent << "CCOSH(" << TYPE << ", " << parameters[0] << ", " << parameters[1] << ")\n";
            break;
        case BH_SINH:
            source << indent << "CSINH(" << TYPE << ", " << parameters[0] << ", " << parameters[1] << ")\n";
            break;
        case BH_TANH:
            source << indent << "CTANH(" << TYPE << ", " << parameters[0] << ", " << parameters[1] << ")\n";
            break;
        case BH_EXP:
            source << indent << "CEXP(" << TYPE << ", " << parameters[0] << ", " << parameters[1] << ")\n";
            break;
        case BH_LOG:
            source << indent << "CLOG(" << parameters[0] << ", " << parameters[1] << ")\n";
            break;
        case BH_LOG10:
            source << indent << "CLOG(" << parameters[0] << ", " << parameters[1] << ")\n";
            source << indent << parameters[0] << " /= log(10.0f);\n";
            break;
        case BH_SQRT:
            source << indent << "CSQRT(" << parameters[0] << ", " << parameters[1] << ")\n";
            break;
        case BH_ABSOLUTE:
            source << indent << "CABS(" << parameters[0] << ", " << parameters[1] << ")\n";
            break;
        case BH_IDENTITY:
            if (type[0] == type[1])
                source << indent << parameters[0] << " = " << parameters[1] << ";\n";
            else if (isComplex(type[1]))
            {
                source << indent << parameters[0] << ".s0 = " << parameters[1] << ".s0;\n";
                source << indent << parameters[0] << ".s1 = " << parameters[1] << ".s1;\n";
            }
            else
            {
                source << indent << parameters[0] << ".s0 = " << parameters[1] << ";\n";
                source << indent << parameters[0] << ".s1 = 0.0" <<
                    (type[0] == OCL_COMPLEX64 ? "f":"") << ";\n";
            }
            break;
        case BH_REAL:
            source << indent << parameters[0] << " = " << parameters[1] << ".s0;\n";
            break;
        case BH_IMAG:
            source << indent << parameters[0] << " = " << parameters[1] << ".s1;\n";
            break;
        default:
            if (resourceManager->verbose())
                std::cerr << "Instruction \"" << bh_opcode_text(opcode) << "\" (" << opcode <<
                    ") not supported for complex operations." << std::endl;
            throw std::runtime_error("Instruction not supported.");
        }
    }
    else // Non complex calculation
    {
        switch(opcode)
        {
        case BH_ADD:
        case BH_ADD_REDUCE:
        case BH_ADD_ACCUMULATE:
            source << indent << parameters[0] << " = " << parameters[1] << " + " << parameters[2] << ";\n";
            break;
        case BH_SUBTRACT:
            source << indent << parameters[0] << " = " << parameters[1] << " - " << parameters[2] << ";\n";
            break;
        case BH_MULTIPLY:
        case BH_MULTIPLY_REDUCE:
        case BH_MULTIPLY_ACCUMULATE:
            source << indent << parameters[0] << " = " << parameters[1] << " * " << parameters[2] << ";\n";
            break;
        case BH_DIVIDE:
            source << indent << parameters[0] << " = " << parameters[1] << " / " << parameters[2] << ";\n";
            break;
        case BH_POWER:
            if (isFloat(type[1]))
                source << indent << parameters[0] << " = pow(" << parameters[1] << ", " << parameters[2] << ");\n";
            else
                source << indent << "IPOW(" << parameters[0] << ", " << parameters[1] << ", " <<
                    parameters[2] << ")\n";
            break;
        case BH_MOD:
            if (isFloat(type[1]))
                source << indent << parameters[0] << " = fmod(" << parameters[1] << ", " <<
                    parameters[2] << ");\n";
            else
                source << indent << parameters[0] << " = " << parameters[1] << " % " << parameters[2] << ";\n";
            break;
        case BH_ABSOLUTE:
            if (isFloat(type[1]))
                source << indent << parameters[0] << " = fabs(" << parameters[1] << ");\n";
            else
                source << indent << parameters[0] << " = abs(" << parameters[1] << ");\n";
            break;
        case BH_RINT:
            source << indent << parameters[0] << " = rint(" << parameters[1] << ");\n";
            break;
        case BH_EXP:
            source << indent << parameters[0] << " = exp(" << parameters[1] << ");\n";
            break;
        case BH_EXP2:
            source << indent << parameters[0] << " = exp2(" << parameters[1] << ");\n";
            break;
        case BH_LOG:
            source << indent << parameters[0] << " = log(" << parameters[1] << ");\n";
            break;
        case BH_LOG10:
            source << indent << parameters[0] << " = log10(" << parameters[1] << ");\n";
            break;
        case BH_EXPM1:
            source << indent << parameters[0] << " = expm1(" << parameters[1] << ");\n";
            break;
        case BH_LOG1P:
            source << indent << parameters[0] << " = log1p(" << parameters[1] << ");\n";
            break;
        case BH_SQRT:
            source << indent << parameters[0] << " = sqrt(" << parameters[1] << ");\n";
            break;
        case BH_SIN:
            source << indent << parameters[0] << " = sin(" << parameters[1] << ");\n";
            break;
        case BH_COS:
            source << indent << parameters[0] << " = cos(" << parameters[1] << ");\n";
            break;
        case BH_TAN:
            source << indent << parameters[0] << " = tan(" << parameters[1] << ");\n";
            break;
        case BH_ARCSIN:
            source << indent << parameters[0] << " = asin(" << parameters[1] << ");\n";
            break;
        case BH_ARCCOS:
            source << indent << parameters[0] << " = acos(" << parameters[1] << ");\n";
            break;
        case BH_ARCTAN:
            source << indent << parameters[0] << " = atan(" << parameters[1] << ");\n";
            break;
        case BH_ARCTAN2:
            source << indent << parameters[0] << " = atan2(" << parameters[1] << ", " << parameters[2] << ");\n";
            break;
        case BH_SINH:
            source << indent << parameters[0] << " = sinh(" << parameters[1] << ");\n";
            break;
        case BH_COSH:
            source << indent << parameters[0] << " = cosh(" << parameters[1] << ");\n";
            break;
        case BH_TANH:
            source << indent << parameters[0] << " = tanh(" << parameters[1] << ");\n";
            break;
        case BH_ARCSINH:
            source << indent << parameters[0] << " = asinh(" << parameters[1] << ");\n";
            break;
        case BH_ARCCOSH:
            source << indent << parameters[0] << " = acosh(" << parameters[1] << ");\n";
            break;
        case BH_ARCTANH:
            source << indent << parameters[0] << " = atanh(" << parameters[1] << ");\n";
            break;
        case BH_BITWISE_AND:
        case BH_BITWISE_AND_REDUCE:
            source << indent << parameters[0] << " = " << parameters[1] << " & " << parameters[2] << ";\n";
            break;
        case BH_BITWISE_OR:
        case BH_BITWISE_OR_REDUCE:
            source << indent << parameters[0] << " = " << parameters[1] << " | " << parameters[2] << ";\n";
            break;
        case BH_BITWISE_XOR:
        case BH_BITWISE_XOR_REDUCE:
            source << indent << parameters[0] << " = " << parameters[1] << " ^ " << parameters[2] << ";\n";
            break;
        case BH_LOGICAL_NOT:
            source << indent << parameters[0] << " = !" << parameters[1] << ";\n";
            break;
        case BH_LOGICAL_AND:
        case BH_LOGICAL_AND_REDUCE:
            source << indent << parameters[0] << " = " << parameters[1] << " && " << parameters[2] << ";\n";
            break;
        case BH_LOGICAL_OR:
        case BH_LOGICAL_OR_REDUCE:
            source << indent << parameters[0] << " = " << parameters[1] << " || " << parameters[2] << ";\n";
            break;
        case BH_LOGICAL_XOR:
        case BH_LOGICAL_XOR_REDUCE:
            source << indent << parameters[0] << " = !" << parameters[1] << " != !" << parameters[2] << ";\n";
            break;
        case BH_INVERT:
            source << indent << parameters[0] << " = ~" << parameters[1] << ";\n";
            break;
        case BH_LEFT_SHIFT:
            source << indent << parameters[0] << " = " << parameters[1] << " << " << parameters[2] << ";\n";
            break;
        case BH_RIGHT_SHIFT:
            source << indent << parameters[0] << " = " << parameters[1] << " >> " << parameters[2] << ";\n";
            break;
        case BH_GREATER:
            source << indent << parameters[0] << " = " << parameters[1] << " > " << parameters[2] << ";\n";
            break;
        case BH_GREATER_EQUAL:
            source << indent << parameters[0] << " = " << parameters[1] << " >= " << parameters[2] << ";\n";
            break;
        case BH_LESS:
            source << indent << parameters[0] << " = " << parameters[1] << " < " << parameters[2] << ";\n";
            break;
        case BH_LESS_EQUAL:
            source << indent << parameters[0] << " = " << parameters[1] << " <= " << parameters[2] << ";\n";
            break;
        case BH_NOT_EQUAL:
            source << indent << parameters[0] << " = " << parameters[1] << " != " << parameters[2] << ";\n";
            break;
        case BH_EQUAL:
            source << indent << parameters[0] << " = " << parameters[1] << " == " << parameters[2] << ";\n";
            break;
        case BH_MAXIMUM:
        case BH_MAXIMUM_REDUCE:
            source << indent << parameters[0] << " = max(" << parameters[1] << ", " << parameters[2] << ");\n";
            break;
        case BH_MINIMUM:
        case BH_MINIMUM_REDUCE:
            source << indent << parameters[0] << " = min(" << parameters[1] << ", " << parameters[2] << ");\n";
            break;
        case BH_IDENTITY:
            source << indent << parameters[0] << " = " << parameters[1] << ";\n";
            break;
        case BH_FLOOR:
            source << indent << parameters[0] << " = floor(" << parameters[1] << ");\n";
            break;
        case BH_CEIL:
            source << indent << parameters[0] << " = ceil(" << parameters[1] << ");\n";
            break;
        case BH_TRUNC:
            source << indent << parameters[0] << " = trunc(" << parameters[1] << ");\n";
            break;
        case BH_LOG2:
            source << indent << parameters[0] << " = log2(" << parameters[1] << ");\n";
            break;
        case BH_ISNAN:
            source << indent << parameters[0] << " = isnan(" << parameters[1] << ");\n";
            break;
        case BH_ISINF:
            source << indent << parameters[0] << " = isinf(" << parameters[1] << ");\n";
            break;
        case BH_RANGE:
            source << indent << parameters[0] << " = " << parameters[1] << ";\n";
            break;
        case BH_RANDOM:
            source << indent << parameters[0] << " = random("  << parameters[1] << ", " << parameters[2] << ");\n";
            break;
        default:
            if (resourceManager->verbose())
                std::cerr << "Instruction \"" << bh_opcode_text(opcode) << "\" (" << opcode <<
                    ") not supported for non complex operations." << std::endl;
            throw std::runtime_error("Instruction not supported.");
        }
    }
}
