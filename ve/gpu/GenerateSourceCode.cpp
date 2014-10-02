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
#include "GenerateSourceCode.hpp"

void generateGIDSource(size_t ndim, std::ostream& source)
{
    assert(ndim > 0);    
    source << "\tconst size_t gidx = get_global_id(0);\n";
    source << "\tif (gidx >= ds0)\n\t\treturn;\n";
    if (ndim > 1)
    {
        source << "\tconst size_t gidy = get_global_id(1);\n";
        source << "\tif (gidy >= ds1)\n\t\treturn;\n";
    }
    if (ndim > 2)
    {
        source << "\tconst size_t gidz = get_global_id(2);\n";
        source << "\tif (gidz >= ds2)\n\t\treturn;\n";
    }
}

void generateOffsetSource(size_t ndim, unsigned int id, std::ostream& source)
{
    assert(ndim > 0);
    for (size_t d = ndim-1; d > 2; --d)
    {
        source << "ids" << d << "*v" << id << "s" << d+1 << " + ";
    }
    if (ndim > 2)
    {
        source << "gidz*v" << id << "s3 + ";
    }
    if (ndim > 1)
    {
        source << "gidy*v" << id << "s2 + ";
    }
    source << "gidx*v" << id << "s1 + v" << id << "s0";
}

#define TYPE ((type[1] == OCL_COMPLEX64) ? "float" : "double")
void generateInstructionSource(const bh_opcode opcode,
                               const std::vector<OCLtype>& type,
                               const std::vector<std::string>& parameters,
                               const std::string& indent,
                               std::ostream& source)
{
    assert(parameters.size() == (size_t)bh_operands(opcode));
    if (isComplex(type[0]) || (type.size()>1 && isComplex(type[1])))
    {

        switch(opcode)
        {
        case BH_ADD:
            source << indent << "CADD(" << parameters[0] << ", " << parameters[1] << ", " << 
                parameters[2] << ")\n";
            break;
        case BH_SUBTRACT:
            source << indent << "CSUB(" << parameters[0] << ", " << parameters[1] << ", " << 
                parameters[2] << ")\n";
            break;
        case BH_MULTIPLY:
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
        case BH_IDENTITY:
            if (isComplex(type[1]))
                source << indent << parameters[0] << " = " << parameters[1] << ";\n";
            else
                source << indent << parameters[0] << ".s0 = " << parameters[1] << ";\n";
            break;
        case BH_REAL:
            source << indent << parameters[0] << " = " << parameters[1] << ".s0;\n";
            break;
        case BH_IMAG:
            source << indent << parameters[0] << " = " << parameters[1] << ".s1;\n";
            break;
        default:
#ifdef DEBUG
            std::cerr << "Instruction \"" << bh_opcode_text(opcode) << "\" not supported." << std::endl;
#endif
            throw std::runtime_error("Instruction not supported.");
        }
    }
    else // Non complex calculation
    {
        switch(opcode)
        {
        case BH_ADD:
            source << indent << parameters[0] << " = " << parameters[1] << " + " << parameters[2] << ";\n";
            break;
        case BH_SUBTRACT:
            source << indent << parameters[0] << " = " << parameters[1] << " - " << parameters[2] << ";\n";
            break;
        case BH_MULTIPLY:
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
            source << indent << parameters[0] << " = " << parameters[1] << " & " << parameters[2] << ";\n";
            break;
        case BH_BITWISE_OR:
            source << indent << parameters[0] << " = " << parameters[1] << " | " << parameters[2] << ";\n";
            break;
        case BH_BITWISE_XOR:
            source << indent << parameters[0] << " = " << parameters[1] << " ^ " << parameters[2] << ";\n";
            break;
        case BH_LOGICAL_NOT:
            source << indent << parameters[0] << " = !" << parameters[1] << ";\n";
            break;
        case BH_LOGICAL_AND:
            source << indent << parameters[0] << " = " << parameters[1] << " && " << parameters[2] << ";\n";
            break;
        case BH_LOGICAL_OR:
            source << indent << parameters[0] << " = " << parameters[1] << " || " << parameters[2] << ";\n";
            break;
        case BH_LOGICAL_XOR:
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
            source << indent << parameters[0] << " = max(" << parameters[1] << ", " << parameters[2] << ");\n";
            break;
        case BH_MINIMUM:
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
            source << indent << parameters[0] << " = gidx;\n";
            break;
        case BH_RANDOM:
            source << indent << parameters[0] << " = random("  << parameters[1] << ", gidx);\n";
            break;
        default:
#ifdef DEBUG
            std::cerr << "Instruction \"" << bh_opcode_text(opcode) << "\" not supported." << std::endl;
#endif
            throw std::runtime_error("Instruction not supported.");
        }
    }
}
