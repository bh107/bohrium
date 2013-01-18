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
#ifndef __BH_INSTRUCTION_H
#define __BH_INSTRUCTION_H

#include "bh_opcode.h"
#include "bh_array.h"
#include "bh_error.h"

#ifdef __cplusplus
extern "C" {
#endif

// Maximum number of operands in a instruction.
#define BH_MAX_NO_OPERANDS (3)

// Datatype header for user-defined functions
/*
    The identifier for the function.
    bh_intp     id;

    Number of output operands
    bh_intp     nout;

    Number of input operands
    bh_intp     nin;

    Total size of the data struct
    bh_intp     struct_size;

    Array of operands (outputs before inputs)
    The macro argument 'nop' specifies the total number of operands
    bh_array*   operand[nop];
*/
#define BH_USER_FUNC_HEADER(nop) \
    bh_intp     id;              \
    bh_intp     nout;            \
    bh_intp     nin;             \
    bh_intp     struct_size;     \
    bh_array*   operand[nop];    \

//The base type for user-defined functions.
typedef struct
{
    BH_USER_FUNC_HEADER(1)
} bh_userfunc;

//Memory layout of the CPHVB instruction
typedef struct
{
    //Instruction status
    bh_error   status;
    //Opcode: Identifies the operation
    bh_opcode  opcode;
    //Id of each operand
    bh_array*  operand[BH_MAX_NO_OPERANDS];
    //Constant included in the instruction (Used if one of the operands == NULL)
    bh_constant constant;
    //Points to the user-defined function when the opcode is BH_USERFUNC.
    bh_userfunc *userfunc;
} bh_instruction;

#ifdef __cplusplus
}
#endif

#endif
