/*
 * Copyright 2011 Troels Blum <troels@blum.dk>
 *
 * This file is part of cphVB.
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
 * along with cphVB.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <iostream>
#include "cphVBinstruction.hpp"

void printInstSpec(cphVBinstruction* inst)
{
    std::cout << cphvb_opcode_text(inst->opcode);
    for (int i = 0; i < cphvb_operands(inst->opcode); ++i)
    {
        if (inst->operand[i] == CPHVB_CONSTANT)
        {
            switch(inst->const_type[i])
            {
            case CPHVB_INT32:
                std::cout << ", " << inst->constant[i].int32;
                break;
            case CPHVB_UINT32:
                std::cout << ", " << inst->constant[i].uint32;
                break;
            case CPHVB_FLOAT32:
                std::cout << ", " << inst->constant[i].float32;
                break;
            default:
                std::cout << ", const";
            }
        }
        else
        {
            std::cout << ", " << inst->operand[i] << "(" << 
                inst->operand[i]->base << ")";
        }
    }
}
