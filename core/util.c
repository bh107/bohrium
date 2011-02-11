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

#include <string.h>
#include <cphvb.h>
#include "private.h"

/* Size needed to store cooresponding serialized instruction
 *
 * @opcode Opcode.
 * @ndim   Number of dimentions.
 * @nc     Number of constants.
 * @return size needed to store cooresponding serialized instruction.
 */
size_t cphvb_size(cphvb_opcode opcode,
                  cphvb_int32 ndim,
                  int nc)
{
    int nops = cphvb_operands(opcode);
    return sizeof(cphvb_opcode) +           //opcode
        sizeof(cphvb_int32) +               //ndim
        sizeof(cphvb_operand) * nops +      //operand[]
        sizeof(cphvb_type) * nops +         //type[]
        sizeof(cphvb_index) * ndim +        //shape[]
        sizeof(cphvb_index) * nops +        //start[]
        sizeof(cphvb_index) * nops * ndim + //stride[][]
        sizeof(cphvb_constant) * nc;        //constant[]

}


/* Number of constants in instruction
 *
 * @inst   Instruction in which number of constants is wanted.
 * @return Number of constants.
 */
int cphvb_constants(const cphvb_instruction* inst)
{
   int res = 0;
   int nops = cphvb_operands(inst->opcode);
   for (int i = 0; i < nops; ++i)
       if (inst->operand[i] == CPHVB_CONSTANT)
           ++res;

   return res;
}


/* Create a new copy of an existing instruction
 *
 * @inst     Instruction to be copied.
 * @newinst  Will be initialized with constants and pointers.
 * @seri     Start of the data area that will contain the new instruction.
 * @return   Pointer to after the data area holding the serialized instruction.
 */
char* cphvb_clone(const cphvb_instruction* inst,
                  cphvb_instruction* newinst,
                  char* seri)
{
    int nc = cphvb_constants(inst);
    int size = cphvb_size(inst->opcode, inst->ndim, nc);
    memcpy(seri,inst->serialized,size);
    newinst->opcode = inst->opcode;
    newinst->ndim = inst->ndim;
    char* res = _setup_pointers(newinst,seri);
    return res + nc;
}

/* Set the shape of an instruction
 *
 * @inst    Instruction to update.
 * @shape[] Shape: number of elements in each dimention.
 */
void cphvb_set_shape(cphvb_instruction* inst,
                     cphvb_index shape[])
{
    for(int i = 0; i < inst->ndim; ++i)
    {
        inst->shape[i] = shape[i];
    }
}


/* Set operand information of an instruction
 *
 * @inst     Instruction to update.
 * @idx      Index of the operand
 * @operand  Id of the operand.
 * @type     Data type of the constant/operand.
 * @start    Start index of the operand.
 * @stride[] Stride in each dimention. If NULL it is ignored.
 *
 */
void cphvb_set_operand(cphvb_instruction* inst,
                       int idx,
                       cphvb_operand operand,
                       cphvb_type type,
                       cphvb_index start,
                       cphvb_index stride[])
{
    int i;
    inst->operand[idx] = operand;
    inst->type[idx] = type;
    inst->start[idx] = start;
    if (stride != NULL)
    {
        for(i = 0; i < inst->ndim; ++i)
        {
            inst->stride[idx][i] = stride[i];
        }
    }
}


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
                        cphvb_type type)
{
    int cidx = 0;
    int i;
    for (i = 0; i < idx; ++i)
        if (inst->operand[i] == CPHVB_CONSTANT)
            ++cidx;

    inst->operand[idx] = CPHVB_CONSTANT;
    // set stride for constant
    for (i = 0; i < inst->ndim; ++i)
        inst->stride[idx][i] = 0;
    inst->start[idx] = 0;
    inst->type[idx] = type;
    inst->constant[cidx] = c;
}
