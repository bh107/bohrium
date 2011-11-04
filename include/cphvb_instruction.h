/*
 * Copyright 2011 Troels Blum <troels@blum.dk>
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

#ifndef __CPHVB_INSTRUCTION_H
#define __CPHVB_INSTRUCTION_H

#include "cphvb_opcode.h"
#include "cphvb_array.h"
#include "cphvb_error.h"

#ifdef __cplusplus
extern "C" {
#endif

// Maximum number of instructions in a batch.
#define CPHVB_MAX_NO_INST (1)

// Operand id used to indicate that the operand is a scalar constant
#define CPHVB_CONSTANT (NULL)

// Maximum number of operands in a instruction.
#define CPHVB_MAX_NO_OPERANDS 3

//Memory layout of the CPHVB instruction
typedef struct
{
    //Instruction status
    cphvb_error status;
    //Opcode: Identifies the operation
    cphvb_opcode   opcode;
    //Id of each operand
    cphvb_array*   operand[CPHVB_MAX_NO_OPERANDS];
    //Constants included in the instruction
    cphvb_constant constant[CPHVB_MAX_NO_OPERANDS];
    //The constant type
    cphvb_type const_type[CPHVB_MAX_NO_OPERANDS];
} cphvb_instruction;

#ifdef __cplusplus
}
#endif

#endif
