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

void generateGIDSource(std::vector<bh_index> shape, std::ostream& source)
{
    size_t ndim = shape.size();
    assert(ndim > 0);    
    if (ndim > 2)
    {
        source << "\tconst size_t gidz = get_global_id(2);\n";
        source << "\tif (gidz >= " << shape[ndim-3] << ")\n\t\treturn;\n";
    }
    if (ndim > 1)
    {
        source << "\tconst size_t gidy = get_global_id(1);\n";
        source << "\tif (gidy >= " << shape[ndim-2] << ")\n\t\treturn;\n";
    }
    source << "\tconst size_t gidx = get_global_id(0);\n";
    source << "\tif (gidx >= " << shape[ndim-1] << ")\n\t\treturn;\n";
}

void generateOffsetSource(const bh_view& operand, std::ostream& source)
{
    bh_index ndim = operand.ndim;
    assert(ndim > 0);
    if (ndim > 2)
    {
        source << "gidz*" << operand.stride[ndim-3] << " + ";
    }
    if (ndim > 1)
    {
        source << "gidy*" << operand.stride[ndim-2] << " + ";
    }
    source << "gidx*" << operand.stride[ndim-1] << " + " << operand.start;
}

#define TYPE ((type[1] == OCL_COMPLEX64) ? "float" : "double")
void generateInstructionSource(bh_opcode opcode,
                               std::vector<OCLtype> type,
                               std::vector<std::string>& parameters, 
                               std::ostream& source)
{
    assert(parameters.size() == (size_t)bh_operands(opcode));
    if (isComplex(type[0]) || (type.size()>1 && isComplex(type[1])))
    {

        switch(opcode)
        {
        case BH_ADD:
            source << "\tCADD(" << parameters[0] << ", " << parameters[1] << ", " << parameters[2] << ")\n";
            break;
        case BH_SUBTRACT:
            source << "\tCSUB(" << parameters[0] << ", " << parameters[1] << ", " << parameters[2] << ")\n";
            break;
        case BH_MULTIPLY:
            source << "\tCMUL(" << parameters[0] << ", " << parameters[1] << ", " << parameters[2] << ")\n";
            break;
        case BH_DIVIDE:
            source << "\tCDIV(" << TYPE << ", " << parameters[0] << ", " << parameters[1] << ", " << 
                parameters[2] << ")\n";
            break;
        case BH_POWER:
            source << "\tCPOW(" << TYPE << ", " << parameters[0] << ", " << parameters[1] << ", " << 
                parameters[2] << ")\n";
            break;
        case BH_EQUAL:
            source << "\tCEQ(" << parameters[0] << ", " << parameters[1] << ", " << parameters[2] << ")\n";
            break;
        case BH_NOT_EQUAL:
            source << "\tCNEQ(" << parameters[0] << ", " << parameters[1] << ", " << parameters[2] << ")\n";
            break;
        case BH_COS:
            source << "\tCCOS(" << TYPE << ", " << parameters[0] << ", " << parameters[1] << ")\n";
            break;
        case BH_SIN:
            source << "\tCSIN(" << TYPE << ", " << parameters[0] << ", " << parameters[1] << ")\n";
            break;
        case BH_TAN:
            source << "\tCTAN(" << TYPE << ", " << parameters[0] << ", " << parameters[1] << ")\n";
            break;
        case BH_COSH:
            source << "\tCCOSH(" << TYPE << ", " << parameters[0] << ", " << parameters[1] << ")\n";
            break;
        case BH_SINH:
            source << "\tCSINH(" << TYPE << ", " << parameters[0] << ", " << parameters[1] << ")\n";
            break;
        case BH_TANH:
            source << "\tCTANH(" << TYPE << ", " << parameters[0] << ", " << parameters[1] << ")\n";
            break;
        case BH_EXP:
            source << "\tCEXP(" << TYPE << ", " << parameters[0] << ", " << parameters[1] << ")\n";
            break;
        case BH_LOG:
            source << "\tCLOG(" << parameters[0] << ", " << parameters[1] << ")\n";
            break;
        case BH_LOG10:
            source << "\tCLOG(" << parameters[0] << ", " << parameters[1] << ")\n";
            source << "\t" << parameters[0] << " /= log(10.0f);\n";
            break;
        case BH_SQRT:
            source << "\tCSQRT(" << parameters[0] << ", " << parameters[1] << ")\n";
            break;
        case BH_IDENTITY:
            if (isComplex(type[1]))
                source << "\t" << parameters[0] << " = " << parameters[1] << ";\n";
            else
                source << "\t" << parameters[0] << ".s0 = " << parameters[1] << ";\n";
            break;
        case BH_REAL:
            source << "\t" << parameters[0] << " = " << parameters[1] << ".s0;\n";
            break;
        case BH_IMAG:
            source << "\t" << parameters[0] << " = " << parameters[1] << ".s1;\n";
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
            source << "\t" << parameters[0] << " = " << parameters[1] << " + " << parameters[2] << ";\n";
            break;
        case BH_SUBTRACT:
            source << "\t" << parameters[0] << " = " << parameters[1] << " - " << parameters[2] << ";\n";
            break;
        case BH_MULTIPLY:
            source << "\t" << parameters[0] << " = " << parameters[1] << " * " << parameters[2] << ";\n";
            break;
        case BH_DIVIDE:
            source << "\t" << parameters[0] << " = " << parameters[1] << " / " << parameters[2] << ";\n";
            break;
        case BH_POWER:
            if (isFloat(type[1]))
                source << "\t" << parameters[0] << " = pow(" << parameters[1] << ", " << parameters[2] << ");\n";
            else
                source << "\t" << parameters[0] << " = pow((float)" << parameters[1] << ", (float)" << parameters[2] << ");\n";   
            break;
        case BH_MOD:
            if (isFloat(type[1]))
                source << "\t" << parameters[0] << " = fmod(" << parameters[1] << ", " << parameters[2] << ");\n";
            else
                source << "\t" << parameters[0] << " = " << parameters[1] << " % " << parameters[2] << ";\n";
            break;
        case BH_ABSOLUTE:
            if (isFloat(type[1]))
                source << "\t" << parameters[0] << " = fabs(" << parameters[1] << ");\n";
            else
                source << "\t" << parameters[0] << " = abs(" << parameters[1] << ");\n";
            break;
        case BH_RINT:
            source << "\t" << parameters[0] << " = rint(" << parameters[1] << ");\n";
            break;
        case BH_EXP:
            source << "\t" << parameters[0] << " = exp(" << parameters[1] << ");\n";
            break;
        case BH_EXP2:
            source << "\t" << parameters[0] << " = exp2(" << parameters[1] << ");\n";
            break;
        case BH_LOG:
            source << "\t" << parameters[0] << " = log(" << parameters[1] << ");\n";
            break;
        case BH_LOG10:
            source << "\t" << parameters[0] << " = log10(" << parameters[1] << ");\n";
            break;
        case BH_EXPM1:
            source << "\t" << parameters[0] << " = expm1(" << parameters[1] << ");\n";
            break;
        case BH_LOG1P:
            source << "\t" << parameters[0] << " = log1p(" << parameters[1] << ");\n";
            break;
        case BH_SQRT:
            source << "\t" << parameters[0] << " = sqrt(" << parameters[1] << ");\n";
            break;
        case BH_SIN:
            source << "\t" << parameters[0] << " = sin(" << parameters[1] << ");\n";
            break;
        case BH_COS:
            source << "\t" << parameters[0] << " = cos(" << parameters[1] << ");\n";
            break;
        case BH_TAN:
            source << "\t" << parameters[0] << " = tan(" << parameters[1] << ");\n";
            break;
        case BH_ARCSIN:
            source << "\t" << parameters[0] << " = asin(" << parameters[1] << ");\n";
            break;
        case BH_ARCCOS:
            source << "\t" << parameters[0] << " = acos(" << parameters[1] << ");\n";
            break;
        case BH_ARCTAN:
            source << "\t" << parameters[0] << " = atan(" << parameters[1] << ");\n";
            break;
        case BH_ARCTAN2:
            source << "\t" << parameters[0] << " = atan2(" << parameters[1] << ", " << parameters[2] << ");\n";
            break;
        case BH_SINH:
            source << "\t" << parameters[0] << " = sinh(" << parameters[1] << ");\n";
            break;
        case BH_COSH:
            source << "\t" << parameters[0] << " = cosh(" << parameters[1] << ");\n";
            break;
        case BH_TANH:
            source << "\t" << parameters[0] << " = tanh(" << parameters[1] << ");\n";
            break;
        case BH_ARCSINH:
            source << "\t" << parameters[0] << " = asinh(" << parameters[1] << ");\n";
            break;
        case BH_ARCCOSH:
            source << "\t" << parameters[0] << " = acosh(" << parameters[1] << ");\n";
            break;
        case BH_ARCTANH:
            source << "\t" << parameters[0] << " = atanh(" << parameters[1] << ");\n";
            break;
        case BH_BITWISE_AND:
            source << "\t" << parameters[0] << " = " << parameters[1] << " & " << parameters[2] << ";\n";
            break;
        case BH_BITWISE_OR:
            source << "\t" << parameters[0] << " = " << parameters[1] << " | " << parameters[2] << ";\n";
            break;
        case BH_BITWISE_XOR:
            source << "\t" << parameters[0] << " = " << parameters[1] << " ^ " << parameters[2] << ";\n";
            break;
        case BH_LOGICAL_NOT:
            source << "\t" << parameters[0] << " = !" << parameters[1] << ";\n";
            break;
        case BH_LOGICAL_AND:
            source << "\t" << parameters[0] << " = " << parameters[1] << " && " << parameters[2] << ";\n";
            break;
        case BH_LOGICAL_OR:
            source << "\t" << parameters[0] << " = " << parameters[1] << " || " << parameters[2] << ";\n";
            break;
        case BH_LOGICAL_XOR:
            source << "\t" << parameters[0] << " = !" << parameters[1] << " != !" << parameters[2] << ";\n";
            break;
        case BH_INVERT:
            source << "\t" << parameters[0] << " = ~" << parameters[1] << ";\n";
            break;
        case BH_LEFT_SHIFT:
            source << "\t" << parameters[0] << " = " << parameters[1] << " << " << parameters[2] << ";\n";
            break;
        case BH_RIGHT_SHIFT:
            source << "\t" << parameters[0] << " = " << parameters[1] << " >> " << parameters[2] << ";\n";
            break;
        case BH_GREATER:
            source << "\t" << parameters[0] << " = " << parameters[1] << " > " << parameters[2] << ";\n";
            break;
        case BH_GREATER_EQUAL:
            source << "\t" << parameters[0] << " = " << parameters[1] << " >= " << parameters[2] << ";\n";
            break;
        case BH_LESS:
            source << "\t" << parameters[0] << " = " << parameters[1] << " < " << parameters[2] << ";\n";
            break;
        case BH_LESS_EQUAL:
            source << "\t" << parameters[0] << " = " << parameters[1] << " <= " << parameters[2] << ";\n";
            break;
        case BH_NOT_EQUAL:
            source << "\t" << parameters[0] << " = " << parameters[1] << " != " << parameters[2] << ";\n";
            break;
        case BH_EQUAL:
            source << "\t" << parameters[0] << " = " << parameters[1] << " == " << parameters[2] << ";\n";
            break;
        case BH_MAXIMUM:
            source << "\t" << parameters[0] << " = max(" << parameters[1] << ", " << parameters[2] << ");\n";
            break;
        case BH_MINIMUM:
            source << "\t" << parameters[0] << " = min(" << parameters[1] << ", " << parameters[2] << ");\n";
            break;
        case BH_IDENTITY:
            source << "\t" << parameters[0] << " = " << parameters[1] << ";\n";
            break;
        case BH_FLOOR:
            source << "\t" << parameters[0] << " = floor(" << parameters[1] << ");\n";
            break;
        case BH_CEIL:
            source << "\t" << parameters[0] << " = ceil(" << parameters[1] << ");\n";
            break;
        case BH_TRUNC:
            source << "\t" << parameters[0] << " = trunc(" << parameters[1] << ");\n";
            break;
        case BH_LOG2:
            source << "\t" << parameters[0] << " = log2(" << parameters[1] << ");\n";
            break;
        case BH_ISNAN:
            source << "\t" << parameters[0] << " = isnan(" << parameters[1] << ");\n";
            break;
        case BH_ISINF:
            source << "\t" << parameters[0] << " = isinf(" << parameters[1] << ");\n";
            break;
        case BH_RANGE:
            source << "\t" << parameters[0] << " = gidx;\n";
            break;
        default:
#ifdef DEBUG
            std::cerr << "Instruction \"" << bh_opcode_text(opcode) << "\" not supported." << std::endl;
#endif
            throw std::runtime_error("Instruction not supported.");
        }
    }
}
