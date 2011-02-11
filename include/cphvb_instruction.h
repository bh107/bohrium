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

#ifndef __CPHVB_INSTRUCTION_H
#define __CPHVB_INSTRUCTION_H

#include "opcode.h"

//Memory layout of the CPHVB instruction
typedef struct
{
    //Opcode: Identifies the operation
    cphvb_opcode   opcode;
    //Id of each operand
    cphvb_operand  operand[CPHVB_MAX_NO_OPERANDS];
    //Constants included in the instruction
    cphvb_constant constant[CPHVB_MAX_NO_OPERANDS];
} cphvb_instruction;

#endif
