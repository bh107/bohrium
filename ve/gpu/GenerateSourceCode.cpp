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

#include <cassert>
#include <stdexcept>
#include "GenerateSourceCode.hpp"

void generateGIDSource(size_t size, std::ostream& source)
{
    if (size > 2)
        source << "\tconst size_t gidz = get_global_id(2);\n";
    if (size > 1)
        source << "\tconst size_t gidy = get_global_id(1);\n";
    source << "\tconst size_t gidx = get_global_id(0);\n";
}

void generateOffsetSource(const cphvb_array* operand, std::ostream& source)
{
    if (operand->ndim > 2)
    {
        source << "gidz*" << operand->stride[2] << " + ";
    }
    if (operand->ndim > 1)
    {
        source << "gidy*" << operand->stride[1] << " + ";
    }
    source << "gidx*" << operand->stride[0] << " + " << operand->start;
}

void generateInstructionSource(cphvb_opcode opcode, 
                               std::vector<std::string>& parameters, 
                               std::ostream& source)
{
    assert(parameters.size() == (size_t)cphvb_operands(opcode));
    switch(opcode)
    {
    case CPHVB_ADD:
        source << "\t" << parameters[0] << " = " << parameters[1] << " + " << parameters[2] << ";\n";
        break;
    case CPHVB_SUBTRACT:
        source << "\t" << parameters[0] << " = " << parameters[1] << " - " << parameters[2] << ";\n";
        break;
    case CPHVB_MULTIPLY:
        source << "\t" << parameters[0] << " = " << parameters[1] << " * " << parameters[2] << ";\n";
        break;
    case CPHVB_DIVIDE:
        source << "\t" << parameters[0] << " = " << parameters[1] << " / " << parameters[2] << ";\n";
        break;
    case CPHVB_NEGATIVE:
        source << "\t" << parameters[0] << " = -" << parameters[1] << ";\n";
        break;
    case CPHVB_SQRT:
        source << "\t" << parameters[0] << " = sqrt(" << parameters[1] << ");\n";
        break;
    case CPHVB_SQUARE:
        source << "\t" << parameters[0] << " = " << parameters[1] << " * " << parameters[1] << ";\n";
        break;
    case CPHVB_BITWISE_AND:
        source << "\t" << parameters[0] << " = " << parameters[1] << " & " << parameters[2] << ";\n";
        break;
    case CPHVB_BITWISE_OR:
        source << "\t" << parameters[0] << " = " << parameters[1] << " | " << parameters[2] << ";\n";
        break;
    case CPHVB_BITWISE_XOR:
        source << "\t" << parameters[0] << " = " << parameters[1] << " ^ " << parameters[2] << ";\n";
        break;
    case CPHVB_LOGICAL_NOT:
        source << "\t" << parameters[0] << " = !" << parameters[1] << ";\n";
        break;
    case CPHVB_LOGICAL_AND:
        source << "\t" << parameters[0] << " = " << parameters[1] << " && " << parameters[2] << ";\n";
        break;
    case CPHVB_LOGICAL_OR:
        source << "\t" << parameters[0] << " = " << parameters[1] << " || " << parameters[2] << ";\n";
        break;
    case CPHVB_LOGICAL_XOR:
        source << "\t" << parameters[0] << " = !" << parameters[1] << " != !" << parameters[2] << ";\n";
        break;
    case CPHVB_INVERT:
        source << "\t" << parameters[0] << " = ~" << parameters[1] << ";\n";
        break;
    case CPHVB_LEFT_SHIFT:
        source << "\t" << parameters[0] << " = " << parameters[1] << " << " << parameters[2] << ";\n";
        break;
    case CPHVB_RIGHT_SHIFT:
        source << "\t" << parameters[0] << " = " << parameters[1] << " >> " << parameters[2] << ";\n";
        break;
    case CPHVB_GREATER:
        source << "\t" << parameters[0] << " = " << parameters[1] << " > " << parameters[2] << ";\n";
        break;
    case CPHVB_GREATER_EQUAL:
        source << "\t" << parameters[0] << " = " << parameters[1] << " >= " << parameters[2] << ";\n";
        break;
    case CPHVB_LESS:
        source << "\t" << parameters[0] << " = " << parameters[1] << " < " << parameters[2] << ";\n";
        break;
    case CPHVB_LESS_EQUAL:
        source << "\t" << parameters[0] << " = " << parameters[1] << " <= " << parameters[2] << ";\n";
        break;
    case CPHVB_NOT_EQUAL:
        source << "\t" << parameters[0] << " = " << parameters[1] << " != " << parameters[2] << ";\n";
        break;
    case CPHVB_EQUAL:
        source << "\t" << parameters[0] << " = " << parameters[1] << " == " << parameters[2] << ";\n";
        break;
    case CPHVB_MAXIMUM:
        source << "\t" << parameters[0] << " = max(" << parameters[1] << ", " << parameters[2] << ");\n";
        break;
    case CPHVB_MINIMUM:
        source << "\t" << parameters[0] << " = min(" << parameters[1] << ", " << parameters[2] << ");\n";
        break;
    case CPHVB_IDENTITY:
        source << "\t" << parameters[0] << " = " << parameters[1] << ";\n";
        break;
    default:
#ifdef DEBUG
        std::cerr << "Instruction \"" << cphvb_opcode_text(opcode) << "\" not supported." << std::endl;
#endif
        throw std::runtime_error("Instruction not supported.");
    }
}
