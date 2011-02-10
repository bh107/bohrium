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

#include "svi.h"
#include "handler.h"
#include "util.h"
#include <stdlib.h>
#include <string.h>

cphvb_error svi_handle_read(cphvb_instruction* inst)
{
    if (inst->operand[1] != CPHVB_CONSTANT || inst->type[1] != CPHVB_PTR)
        return CPHVB_TYPE_NOT_SUPPORTED_BY_OP;
    
    void* arrp;
    if (cphvb_is_continuous(inst->ndim, inst->shape, inst->stride[0]))
    {
        
        cphvb_error res = svi_get_array_pointer(inst->operand[0], &arrp);
        if (res != CPHVB_SUCCESS)
            return res;
        *(inst->constant[0].ptr) = arrp + inst->start[0] * 
                                          cphvb_operands(inst->type[0]);
        return CPHVB_SUCCESS;
    }
    
    // TODO: Handling of non continuous request
    return CPHVB_INST_NOT_SUPPORTED_FOR_SLICE;
}
