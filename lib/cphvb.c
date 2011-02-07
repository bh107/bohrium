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

#include <stdlib.h>
#include <string.h>
#include "cphvb.h"
#include "private.h"

/* Setup the pointers in cphvb_instruction struct for seri
 *
 * @inst   Will be initialized with pointers
 * @seri   Start of the data area that contains the instruction
 * @return Pointer to after stride[][].
 */
char* _setup_pointers(cphvb_instruction* inst, 
                      char* seri)
{
    int i;
    int nops = cphvb_operands[inst->opcode];
    inst->serialized = seri;
    inst->operand = (cphvb_operand*)(seri += sizeof(cphvb_opcode) + 
                                     sizeof(cphvb_int32));
    inst->type = (cphvb_type*)(seri += sizeof(cphvb_operand) * nops);
    inst->shape = (cphvb_index*)(seri += sizeof(cphvb_type) * nops);
    inst->start = (cphvb_index*)(seri += sizeof(cphvb_index) * inst->ndim);
    seri += sizeof(cphvb_index) * nops;
    for (i = 0; i < nops; ++i)
    {
        inst->stride[i] = (cphvb_index*)seri;
        seri += sizeof(cphvb_index) * inst->ndim; 
    }
    inst->constant = (cphvb_constant*)seri;
    return seri;
}

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
                 char* seri)
{
    inst->opcode = opcode;
    inst->ndim = ndim;
    *(cphvb_opcode*)seri = opcode;
    *(cphvb_int32*)(seri + sizeof(cphvb_opcode)) = ndim;    
    return _setup_pointers(inst,seri) + sizeof(cphvb_constant) * nc;
}


/* Restore an instruction from its serialized (raw) format 
 *
 * @inst   Will be initialized with constants and pointers.
 * @seri   Start of the data area that contains the serialized instruction.
 * @return Pointer to after the data area holding the serialized instruction.
 */
char* cphvb_restore(cphvb_instruction* inst, 
                    const char* seri)
{
    inst->opcode = *(cphvb_opcode*)seri;
    inst->ndim = *(cphvb_int32*)(seri + sizeof(cphvb_opcode));
    char* res = _setup_pointers(inst,(char*)seri);
    return res + sizeof(cphvb_constant) * cphvb_constants(inst);
}
