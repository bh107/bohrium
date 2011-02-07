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
    return res + cphvb_constants(inst);
}
