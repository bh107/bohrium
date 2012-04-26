/*
 * Copyright 2011 Simon A. F. Lund <safl@safl.dk>
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
#include <cphvb.h>
#include "cphvb_ve_score.h"
#include <assert.h>
#include <cphvb_comp.h>

static cphvb_com *myself = NULL;
static cphvb_userfunc_impl reduce_impl = NULL;
static cphvb_intp reduce_impl_id = 0;
static cphvb_userfunc_impl random_impl = NULL;
static cphvb_intp random_impl_id = 0;

cphvb_error cphvb_ve_score_init(cphvb_com *self)
{
    myself = self;

    return CPHVB_SUCCESS;
}

cphvb_error cphvb_ve_score_execute(

    cphvb_intp          instruction_count,
    cphvb_instruction*   instruction_list

) {
    cphvb_intp count=0, nops, i;

    while(count < instruction_count)
    {
        cphvb_instruction* inst = &instruction_list[count];

        // Make sure that the array memory is allocated.
        nops = cphvb_operands(inst->opcode);
        for(i = 0; i<nops; i++) {

            if (!cphvb_is_constant(inst->operand[i])) {
                if (cphvb_data_malloc(inst->operand[i]) != CPHVB_SUCCESS) {
                    return CPHVB_OUT_OF_MEMORY;
                }
            }

        }

        // Execute it...
        switch(inst->opcode)
        {
            case CPHVB_NONE:
            case CPHVB_DISCARD:
            case CPHVB_SYNC:
                break;
            case CPHVB_USERFUNC:

                if(inst->userfunc->id == reduce_impl_id)
                {
                    inst->status = reduce_impl(inst->userfunc, NULL);
                }
                else if(inst->userfunc->id == random_impl_id)
                {
                    inst->status = random_impl(inst->userfunc, NULL);
                }
                else// Unsupported instruction
                {
                    inst->status = CPHVB_TYPE_NOT_SUPPORTED;
                }

            default://This is a regular operation.
                if(inst->status == CPHVB_INST_UNDONE) {
                    cphvb_comp_apply( inst );
                }
        }
        ++count;

    }

    return CPHVB_SUCCESS;

}

cphvb_error cphvb_ve_score_shutdown( void )
{
    return CPHVB_SUCCESS;
}

cphvb_error cphvb_ve_score_reg_func(char *lib, char *fun, cphvb_intp *id) {

    if(strcmp("cphvb_reduce", fun) && reduce_impl == NULL)
    {
        cphvb_com_get_func(myself, lib, fun, &reduce_impl);
        reduce_impl_id = *id;
    }
    else if(strcmp("cphvb_random", fun) && random_impl == NULL)
    {
        cphvb_com_get_func(myself, lib, fun, &random_impl);
        random_impl_id = *id;
    }
    return CPHVB_SUCCESS;
}

//Implementation of the user-defined funtion "reduce". Note that we
//follow the function signature defined by cphvb_userfunc_impl.
cphvb_error cphvb_reduce(cphvb_userfunc *arg, void* ve_arg)
{
    cphvb_reduce_type *a = (cphvb_reduce_type *) arg;
    cphvb_instruction tinst;
    cphvb_instruction *inst[1] = {&tinst};
    cphvb_error err;
    cphvb_intp i,j;
    cphvb_index coord[CPHVB_MAXDIM];
    memset(coord, 0, a->operand[1]->ndim * sizeof(cphvb_index));

    if(cphvb_operands(a->opcode) != 3)
    {
        fprintf(stderr, "Reduce only support binary operations.\n");
        exit(-1);
    }

    //Make sure that the array memory is allocated.
    if(cphvb_data_malloc(a->operand[0]) != CPHVB_SUCCESS ||
       cphvb_data_malloc(a->operand[1]) != CPHVB_SUCCESS)
    {
        return CPHVB_OUT_OF_MEMORY;
    }

    //We need a tmp copy of the arrays.
    cphvb_array *out = a->operand[0];
    cphvb_array *in  = a->operand[1];
    cphvb_array tmp  = *in;
    tmp.base = cphvb_base_array(in);
    cphvb_intp step = in->stride[a->axis];
    tmp.start = 0;
    j=0;
    for(i=0; i<in->ndim; ++i)//Remove the 'axis' dimension from in
        if(i != a->axis)
        {
            tmp.shape[j] = in->shape[i];
            tmp.stride[j] = in->stride[i];
            ++j;
        }
    --tmp.ndim;

    //We copy the first element to the output.
    inst[0]->status = CPHVB_INST_UNDONE;
    inst[0]->opcode = CPHVB_IDENTITY;
    inst[0]->operand[0] = out;
    inst[0]->operand[1] = &tmp;
    err = cphvb_comp_apply( inst[0] );    // execute the instruction...
    if(err != CPHVB_SUCCESS)
        return err;
    tmp.start += step;

    //Reduce over the 'axis' dimension.
    //NB: the first element is already handled.
    inst[0]->status = CPHVB_INST_UNDONE;
    inst[0]->opcode = a->opcode;
    inst[0]->operand[0] = out;
    inst[0]->operand[1] = out;
    inst[0]->operand[2] = &tmp;
    cphvb_intp axis_size = in->shape[a->axis];

    for(i=1; i<axis_size; ++i)
    {
        //One block per thread.
        err = cphvb_comp_apply(inst[0]);
        if(err != CPHVB_SUCCESS)
            return err;
        tmp.start += step;
    }

    return CPHVB_SUCCESS;
}
