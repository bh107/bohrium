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
#include "bundler.hpp"
#include "dispatch.h"
#include <assert.h>
#include "get_traverse.hpp"


static cphvb_com *myself = NULL;
static cphvb_userfunc_impl reduce_impl = NULL;
static cphvb_intp reduce_impl_id = 0;
static cphvb_userfunc_impl random_impl = NULL;
static cphvb_intp random_impl_id = 0;
static cphvb_intp nblocks = 16;


cphvb_error cphvb_ve_score_init(cphvb_com *self)
{
    myself = self;

    char *env = getenv("CPHVB_SCORE_NBLOCKS");
    if(env != NULL)
        nblocks = atoi(env);

    return CPHVB_SUCCESS;
}

cphvb_error cphvb_ve_score_execute(

    cphvb_intp          instruction_count,
    cphvb_instruction*   instruction_list

) {
    cphvb_intp count=0;
    cphvb_instruction* regular_inst[CPHVB_MAX_NO_INST];

    while(count < instruction_count)
    {
        cphvb_instruction* inst = &instruction_list[count];
        //Handle user-defined operations.
        if(inst->opcode == CPHVB_USERFUNC)
        {
            if(inst->userfunc->id == reduce_impl_id)
            {
                inst->status = reduce_impl(inst->userfunc);
            }
            else if(inst->userfunc->id == random_impl_id)
            {
                inst->status = random_impl(inst->userfunc);
            }
            else// Unsupported instruction
            {
                inst->status = CPHVB_TYPE_NOT_SUPPORTED;
            }
            ++count;
        }

        //Gather regular operations.
        cphvb_intp regular_size=0;
        while(count < instruction_count)
        {
            inst = &instruction_list[count];
            if(inst->opcode == CPHVB_USERFUNC)
                break;

            switch(inst->opcode)
            {
                case CPHVB_NONE:
                case CPHVB_DISCARD:
                case CPHVB_RELEASE:
                case CPHVB_SYNC:
                    break;
                default://This is a regular operation.
                    if(inst->status == CPHVB_INST_UNDONE)
                        regular_inst[regular_size++] = inst;
            }
            ++count;
        }
        //Dispatch the regular operations.
        cphvb_instruction** cur_bundle = regular_inst;
        while(regular_size > 0)
        {
            //Get number of consecutive bundeable instruction.
            cphvb_intp bundle_size = bundle(cur_bundle, regular_size);
            //Dispatch the bundle
            dispatch_bundle(cur_bundle, bundle_size, nblocks);
            //Iterate to next bundle.
            regular_size -= bundle_size;
            cur_bundle += bundle_size;
            assert(regular_size >= 0);
        }
    }

    return CPHVB_SUCCESS;

}

cphvb_error cphvb_ve_score_shutdown( void ) {

    return CPHVB_SUCCESS;

}

cphvb_error cphvb_ve_score_reg_func(char *lib, char *fun, cphvb_intp *id) {

    if(reduce_impl == NULL)//We only support one user-defind function
    {
        cphvb_com_get_func(myself, lib, fun, &reduce_impl);
        reduce_impl_id = *id;
    }
    return CPHVB_SUCCESS;
}

//Implementation of the user-defined funtion "reduce". Note that we
//follows the function signature defined by cphvb_userfunc_impl.
cphvb_error cphvb_reduce(cphvb_userfunc *arg)
{
    cphvb_reduce_type *a = (cphvb_reduce_type *) arg;
    cphvb_instruction inst;
    cphvb_intp i,j;
    cphvb_index coord[CPHVB_MAXDIM];
    memset(coord, 0, a->operand[1]->ndim * sizeof(cphvb_index));

    if(cphvb_operands(a->opcode) != 3)
    {
        fprintf(stderr, "Reduce only support binary operations.\n");
        exit(-1);
    }

    //Make sure that the array memory is allocated.
    if(cphvb_malloc_array_data(a->operand[0]) != CPHVB_SUCCESS ||
       cphvb_malloc_array_data(a->operand[1]) != CPHVB_SUCCESS)
    {
        return CPHVB_OUT_OF_MEMORY;
    }

    //We need a tmp copy of the arrays.
    cphvb_array *out = a->operand[0];
    cphvb_array *in  = a->operand[1];
    cphvb_array tmp  = *in;
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
    inst.status = CPHVB_INST_UNDONE;
    inst.opcode = CPHVB_IDENTITY;
    inst.operand[0] = out;
    inst.operand[1] = &tmp;
    traverse_ptr trav = get_traverse(&inst);
    if(trav == NULL)
        return CPHVB_ERROR;
    cphvb_error err = trav(&inst);
    if(err != CPHVB_SUCCESS)
        return err;
    tmp.start += step;

    //Reduce over the 'axis' dimension.
    //NB: the first element is already handled.
    inst.status = CPHVB_INST_UNDONE;
    inst.opcode = a->opcode;
    inst.operand[0] = out;
    inst.operand[1] = out;
    inst.operand[2] = &tmp;
    cphvb_intp axis_size = in->shape[a->axis];
    trav = get_traverse(&inst);
    if(trav == NULL)
        return CPHVB_ERROR;
    for(i=1; i<axis_size; ++i)
    {
        err = trav(&inst);
        if(err != CPHVB_SUCCESS)
            return err;
        tmp.start += step;
    }

    return CPHVB_SUCCESS;
}
