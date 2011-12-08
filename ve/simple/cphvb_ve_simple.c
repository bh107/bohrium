/*
 * Copyright 2011 Mads R. B. Kristensen <madsbk@gmail.com>
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
 * along with cphVB. If not, see <http://www.gnu.org/licenses/>.
 */

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <cphvb.h>

static cphvb_com *self = NULL;
static cphvb_userfunc_impl reduce_impl = NULL;
static cphvb_intp reduce_impl_id = 0;

cphvb_error cphvb_ve_simple_init(cphvb_com *new_self)
{
    self = new_self;
    return CPHVB_SUCCESS;
}

cphvb_error cphvb_ve_simple_shutdown(void)
{
    return CPHVB_SUCCESS;
}

cphvb_error cphvb_ve_simple_reg_func(char *lib, char *fun, cphvb_intp *id)
{
    if(reduce_impl == NULL)//We only support one user-defind function
    {
        cphvb_com_get_func(self, lib, fun, &reduce_impl);
        reduce_impl_id = *id;
    }
    return CPHVB_SUCCESS;
}

cphvb_error cphvb_ve_simple_execute(cphvb_intp instruction_count,
                                    cphvb_instruction instruction_list[])
{
    cphvb_intp i;
    for(i=0; i<instruction_count; ++i)
    {
        cphvb_instruction *inst = &instruction_list[i];
        if(inst->status != CPHVB_INST_UNDONE)
            break;//We should only handle undone instructions.

        switch(inst->opcode)
        {
        case CPHVB_NONE:
        case CPHVB_DISCARD:
        case CPHVB_RELEASE:
        case CPHVB_SYNC:
            //Nothing to do since we only use main memory.
            break;
        case CPHVB_ADD:
        {
            cphvb_intp j, notfinished=1;
            cphvb_float32 *d0, *d1, *d2;
            cphvb_array *a0 = inst->operand[0];
            cphvb_array *a1 = inst->operand[1];
            cphvb_array *a2 = inst->operand[2];
            cphvb_index coord[CPHVB_MAXDIM];
            memset(coord, 0, CPHVB_MAXDIM * sizeof(cphvb_index));

            if(cphvb_malloc_array_data(a0) != CPHVB_SUCCESS)
            {
                inst->status = CPHVB_OUT_OF_MEMORY;
                return CPHVB_PARTIAL_SUCCESS;
            }
            if(cphvb_malloc_array_data(a1) != CPHVB_SUCCESS)
            {
                inst->status = CPHVB_OUT_OF_MEMORY;
                return CPHVB_PARTIAL_SUCCESS;
            }
            if(cphvb_malloc_array_data(a2) != CPHVB_SUCCESS)
            {
                inst->status = CPHVB_OUT_OF_MEMORY;
                return CPHVB_PARTIAL_SUCCESS;
            }
            d0 = cphvb_base_array(inst->operand[0])->data;
            d1 = cphvb_base_array(inst->operand[1])->data;
            d2 = cphvb_base_array(inst->operand[2])->data;

            //We only support float32
            if(cphvb_type_operand(inst, 0) != CPHVB_FLOAT32 ||
               cphvb_type_operand(inst, 1) != CPHVB_FLOAT32 ||
               cphvb_type_operand(inst, 2) != CPHVB_FLOAT32)
            {
                inst->status = CPHVB_TYPE_NOT_SUPPORTED;
                return CPHVB_PARTIAL_SUCCESS;
            }

            while(notfinished)
            {
                cphvb_intp off0=0, off1=0, off2=0;

                for(j=0; j<a0->ndim; ++j)
                    off0 += coord[j] * a0->stride[j];
                off0 += a0->start;
                for(j=0; j<a0->ndim; ++j)
                    off1 += coord[j] * a1->stride[j];
                off1 += a1->start;
                for(j=0; j<a0->ndim; ++j)
                    off2 += coord[j] * a2->stride[j];
                off2 += a2->start;

                //Compute the element.
                *(d0 + off0) = *(d1 + off1) + *(d2 + off2);

                //Iterate coord one element.
                for(j=a0->ndim-1; j >= 0; j--)
                {
                    if(++coord[j] >= a0->shape[j])
                    {
                        //We are finished, if wrapping around.
                        if(j == 0)
                        {
                            notfinished = 0;
                            break;
                        }
                        coord[j] = 0;
                    }
                    else
                        break;
                }
            }
            break;
        }

        case CPHVB_USERFUNC:
            if(inst->userfunc->id == reduce_impl_id)
            {
                reduce_impl(inst->userfunc);
                break;
            }//Else we don't know it and go to default.
        default:
            inst->status = CPHVB_INST_NOT_SUPPORTED;
            return CPHVB_PARTIAL_SUCCESS;
        }
    }

    return CPHVB_SUCCESS;
}

//Implementation of the user-defined funtion "reduce". Note that we
//follows the function signature defined by cphvb_userfunc_impl.
cphvb_error cphvb_reduce(cphvb_userfunc *arg)
{
    printf("cphvb_reduce\n");
}


