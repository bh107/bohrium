/*
This file is part of cphVB and copyright (c) 2012 the cphVB team:
http://cphvb.bitbucket.org

cphVB is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as 
published by the Free Software Foundation, either version 3 
of the License, or (at your option) any later version.

cphVB is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the 
GNU Lesser General Public License along with cphVB. 

If not, see <http://www.gnu.org/licenses/>.
*/
#ifndef __CPHVB_INSTRUCTION_H
#define __CPHVB_INSTRUCTION_H

#include "cphvb_opcode.h"
#include "cphvb_array.h"
#include "cphvb_error.h"

#ifdef __cplusplus
extern "C" {
#endif

// Maximum number of operands in a instruction.
#define CPHVB_MAX_NO_OPERANDS (3)

// Datatype header for user-defined functions
/*
    The identifier for the function.
    cphvb_intp     id;

    Number of output operands
    cphvb_intp     nout;

    Number of input operands
    cphvb_intp     nin;

    Total size of the data struct
    cphvb_intp     struct_size;

    Array of operands (outputs before inputs)
    The macro argument 'nop' specifies the total number of operands
    cphvb_array*   operand[nop];
*/
#define CPHVB_USER_FUNC_HEADER(nop) \
    cphvb_intp     id;              \
    cphvb_intp     nout;            \
    cphvb_intp     nin;             \
    cphvb_intp     struct_size;     \
    cphvb_array*   operand[nop];    \

//The base type for user-defined functions.
typedef struct
{
    CPHVB_USER_FUNC_HEADER(1)
} cphvb_userfunc;

//Memory layout of the CPHVB instruction
typedef struct
{
    //Instruction status
    cphvb_error   status;
    //Opcode: Identifies the operation
    cphvb_opcode  opcode;
    //Id of each operand
    cphvb_array*  operand[CPHVB_MAX_NO_OPERANDS];
    //Constant included in the instruction (Used if one of the operands == NULL)
    cphvb_constant constant;
    //Points to the user-defined function when the opcode is CPHVB_USERFUNC.
    cphvb_userfunc *userfunc;
} cphvb_instruction;

#ifdef __cplusplus
}
#endif

#endif
