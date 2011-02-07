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
#include "cast.h"
#include "util.h"
#include <stdio.h>

cphvb_error svi_add_int32(cphvb_instruction* inst)
{
    // Check that we can convert to the result type
    if (!svi_can_cast_to_int32(inst->type[1]))
        return CPHVB_TYPE_COMBINATION_NOT_SUPPORTED;
    if (!svi_can_cast_to_int32(inst->type[2]))
        return CPHVB_TYPE_COMBINATION_NOT_SUPPORTED;
    
    // Check that all arrays are known
    // and take care of constants
    cphvb_error res;    
    cphvb_int32* result;
    res = svi_get_array_pointer(inst->operand[0],(void**)&result);
    if (res != CPHVB_SUCCESS)
        return res;
    void* arg1;
    int cidx = 0;
    if (inst->operand[1] == CPHVB_CONSTANT)
    {
        arg1 = &inst->constant[cidx++];
    } else
    {
        res = svi_get_array_pointer(inst->operand[1],&arg1);
        if (res != CPHVB_SUCCESS)
            return res;
    } 
    
    void* arg2;
    if (inst->operand[2] == CPHVB_CONSTANT)
    {
        arg2 = &inst->constant[cidx++];
    } else
    {
        res = svi_get_array_pointer(inst->operand[2],&arg2);
        if (res != CPHVB_SUCCESS)
            return res;
    } 

    // now to the calcutations
    int off0, off1, off2, i;
    cphvb_int32 iarg1, iarg2;
    for (i = 0; i < cphvb_nelements(inst->ndim,inst->shape); ++i)
    {
        off0 = cphvb_calc_offset(inst->ndim, inst->shape, inst->stride[0], i);
        off1 = cphvb_calc_offset(inst->ndim, inst->shape, inst->stride[1], i);
        off2 = cphvb_calc_offset(inst->ndim, inst->shape, inst->stride[2], i);
        iarg1 = svi_to_int32(arg1 + off1 * cphvb_typesize[inst->type[1]], 
                          inst->type[1]);
        iarg2 = svi_to_int32(arg2 + off2 * cphvb_typesize[inst->type[2]], 
                                     inst->type[2]);
        *(result + off0) = iarg1 + iarg2;
    }
    return CPHVB_SUCCESS;
}


cphvb_error svi_handle_add(cphvb_instruction* inst)
{
    switch(inst->type[0])
    {
    case CPHVB_INT32:
        return svi_add_int32(inst);
    default:
        return CPHVB_TYPE_NOT_SUPPORTED;
    }
}

