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

#ifndef __CPHVB_H
#define __CPHVB_H

#include <stdint.h>
#include <stddef.h>
#include <stdarg.h>

#include "opcode.h"
#include "error.h"
#include "type.h"

// Operand id used to indicate that the operand is a scalar constant
#define CPHVB_CONSTANT -1

// Data type for content of the CPHVB-instruction struct
typedef cphvb_int32 cphvb_operand;
typedef cphvb_int64 cphvb_index;
typedef cphvb_int32 cphvb_opcode;
typedef cphvb_int32 cphvb_type;

/* Momory layout of the CPHVB instruction code data block
 *
 * opcode             //Opcode: Identifies the operation            
 * ndim               //Number of dimentions                         
 * operand[nops]      //Id of each operand                           
 * type[nops]         //The type of data in each operand      
 * shape[ndim]        //Number of elements in each dimention         
 * start[nops]        //Index of start element for each operand 
 * stride[nops][ndim] //The stride for each dimention per array      
 * constant[?]        //The constants included in the instruction
 *                    // as indicated by operand == CPHVB_CONSTANT      
 *                            
 * nops is the number of operands. Discribed by the opcode
 */

typedef struct 
{
    cphvb_opcode    opcode;    //Opcode: Identifies the operation
    cphvb_int32     ndim;      //Number of dimentions
    cphvb_operand*  operand;   //Id of each operand                           
    cphvb_type*     type;      //The type of data in each operand
    cphvb_index*    shape;     //Number of elements in each dimention         
    cphvb_index*    start;     //Index of start element for each operand 
    cphvb_index*    stride[CPHVB_MAX_NO_OPERANDS]; 
    cphvb_constant* constant;  //Constants included in the instruction
    char*         serialized;  //The raw data that reprecents the instruction
} cphvb_instruction;

/* Initialize a new instruction
 *
 * @inst   Will be initialized with constants and pointers.
 * @opcode Opcode.
 * @ndim   Number of dimentions.
 * @nc     Number of constants.
 * @seri   Start of the data area that will contain the serialized instruction.
 * @return Pointer to after the data area holding the serialized instruction.
 */
char* cphvb_init(cphvb_instruction* inst, 
                 cphvb_opcode opcode, 
                 cphvb_int32 ndim, 
                 int nc,
                 char* seri);


/* Restore an instruction from its serialized (raw) format 
 *
 * @inst   Will be initialized with constants and pointers.
 * @seri   Start of the data area that contains the serialized instruction.
 * @return Pointer to after the data area holding the serialized instruction.
 */
char* cphvb_restore(cphvb_instruction* inst, 
                    const char* seri);


/* Number of constants in instruction
 *
 * @inst   Instruction in which number of constants is wanted.
 * @return Number of constants.
 */
int cphvb_constants(const cphvb_instruction* inst);


/* Size needed to store cooresponding serialized instruction
 *
 * @opcode Opcode.
 * @ndim   Number of dimentions.
 * @nc     Number of constants.
 * @return size needed to store cooresponding serialized instruction.
 */
size_t cphvb_size(cphvb_opcode opcode, 
                  cphvb_int32 ndim, 
                  int nc);


/* Create a new copy of an existing instruction
 *
 * @inst     Instruction to be copied.
 * @newinst  Will be initialized with constants and pointers.
 * @seri     Start of the data area that will contain the new instruction.
 * @return   Pointer to after the data area holding the serialized instruction.
 */
char* cphvb_clone(const cphvb_instruction* inst,
                  cphvb_instruction* newinst,
                  char* seri);


/* Set the shape of an instruction
 *
 * @inst    Instruction to update.
 * @shape[] Shape: number of elements in each dimention.
 */
void cphvb_set_shape(cphvb_instruction* inst, 
                     cphvb_index shape[]);


/* Set operand information of an instruction
 *
 * @inst     Instruction to update.
 * @idx      Index of the operand
 * @operand  Id of the operand.
 * @type     Data type of the constant/operand.
 * @start    Start index of the operand.
 * @stride[] Stride in each dimention.
 *           
 */
void cphvb_set_operand(cphvb_instruction* inst,
                       int idx,
                       cphvb_operand operand,
                       cphvb_type type,
                       cphvb_index start,
                       cphvb_index stride[]);


/* Sets a constant operand in CPHVB oparation. 
 *
 * NOTE: Operands have to be set in accending order, when using this 
 * function.
 *
 * @inst   Instruction to update.
 * @idx    Index of operand.
 * @c      The constant.
 * @type   Data type of the constant/operand.
*/
void cphvb_set_constant(cphvb_instruction* inst, 
                        int idx, 
                        cphvb_constant c, 
                        cphvb_type type);


/* Pretty print instruction
 *
 * Mainly for debugging purposes.
 *
 * @inst   Instruction to print.
 * @size   Write at most this many bytes to buffer.
 * @buf    Buffer to contain the string.
 * @return Number of characters printed.
 */
int cphvb_snprint(const cphvb_instruction* inst, 
                  size_t size, 
                  char* buf);

#endif
