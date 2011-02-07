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
#include "svi.h"
#include "handler.h"
#include <stdlib.h>
#include <stdio.h>

#define SVI_QUEUE_SIZE 1024

bool svi_initialized = FALSE;
cphvb_instruction* svi_instruction_queue;


svi_callback my_callback;

cphvb_error svi_init(svi_callback callback)
{
    if (svi_initialized)
        return CPHVB_ALREADY_INITALIZED;
    my_callback = callback;
    svi_operand_map = calloc(SVI_MAPSIZE,sizeof(void*));
    svi_instruction_queue = (cphvb_instruction*)malloc(
        sizeof(cphvb_instruction) * SVI_QUEUE_SIZE);
    if(svi_operand_map == NULL || svi_instruction_queue == NULL)
    {
        return CPHVB_ERROR;
    }
    else {
        svi_initialized = TRUE;
        return CPHVB_SUCCESS;
    }
    
}

cphvb_error svi_handle_inst(cphvb_instruction* inst)
{
    cphvb_error res;
    switch(inst->opcode)
    {
    case CPHVB_ADD:
        res = svi_handle_add(inst);
        break;
    case CPHVB_MALLOC:
        res = svi_handle_malloc(inst);
        break;
    case CPHVB_FREE:
        res = svi_handle_free(inst);
        break;
    case CPHVB_READ:
        res = svi_handle_read(inst);
        break;
    default:
        res = CPHVB_INST_NOT_SUPPORTED;
    }
    return res;
}

cphvb_error svi_do(cphvb_instruction* inst)
{
#ifdef SVI_DEBUG
    char buf[1024];
    cphvb_snprint(inst,1024,buf);
    printf("[SVI] Executing: %s",buf);
#endif
    if (inst->operand[0] == CPHVB_CONSTANT)
        return CPHVB_RESULT_IS_CONSTANT;
    
    return svi_handle_inst(inst);
}



cphvb_error svi_execute(cphvb_int32 batch_id,
                        cphvb_int32 instruction_count,
                        char* seri)
{
    cphvb_error res;
    
    if (!svi_initialized)
        return CPHVB_NOT_INITALIZED;
    
    int i;
    for (i = 0; i < instruction_count; ++i)
    {
        seri = cphvb_restore(&svi_instruction_queue[i], seri);
        res = svi_do(&svi_instruction_queue[i]);
        if (res != CPHVB_SUCCESS)
        {
            (*my_callback)(batch_id,i,res);
            return res;
        }
        
    }
    (*my_callback)(batch_id,i,CPHVB_SUCCESS);
    
    return CPHVB_SUCCESS;
}
