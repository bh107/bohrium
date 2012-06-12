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
#include "cphvb_ve_mcore.h"
#include "dispatch.h"
#include <assert.h>

static cphvb_component *myself = NULL;
static cphvb_userfunc_impl reduce_impl = NULL;
static cphvb_intp reduce_impl_id = 0;
static cphvb_userfunc_impl random_impl = NULL;
static cphvb_intp random_impl_id = 0;
static cphvb_intp nblocks = 16;

static cphvb_intp cphvb_ve_mcore_buffersizes = 0;
static cphvb_instruction** cphvb_ve_mcore_instruction_list = NULL;

cphvb_error cphvb_ve_mcore_init(cphvb_component *self)
{
    myself = self;
    char *env = getenv("CPHVB_MCORE_NBLOCKS");
    if(env != NULL)
        nblocks = atoi(env);
    if(nblocks <= 0)
    {
        fprintf(stderr, "CPHVB_MCORE_NBLOCKS should be greater than"
                        " zero!\n");
        return CPHVB_ERROR;
    }
    //Initiate the dispatcher.
    dispatch_init();

    return CPHVB_SUCCESS;
}

cphvb_error cphvb_ve_mcore_execute(

    cphvb_intp          instruction_count,
    cphvb_instruction*   instruction_list

) {
    cphvb_intp count=0;
    
    //Make sure we have enough space
    if (instruction_count > cphvb_ve_mcore_buffersizes) 
    {
    	cphvb_intp mcount = instruction_count;

    	//We only work in multiples of 1000
    	if (mcount % 1000 != 0)
    		mcount = (mcount + 1000) - (mcount % 1000);
    		
    	//Make sure we re-allocate on error
    	cphvb_ve_mcore_buffersizes = 0;
    	
    	if (cphvb_ve_mcore_instruction_list != NULL) {
    		free(cphvb_ve_mcore_instruction_list);
    		cphvb_ve_mcore_instruction_list = NULL;
    	}

    	cphvb_ve_mcore_instruction_list = (cphvb_instruction**)malloc(sizeof(cphvb_instruction*) * mcount);
    	
    	if (cphvb_ve_mcore_instruction_list == NULL)
    		return CPHVB_OUT_OF_MEMORY;
    	
    	cphvb_ve_mcore_buffersizes = mcount;
    }

	cphvb_instruction** regular_inst = cphvb_ve_mcore_instruction_list;

    while(count < instruction_count)
    {
        cphvb_instruction* inst = &instruction_list[count];
        //Handle user-defined operations.
        if(inst->opcode == CPHVB_USERFUNC)
        {
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
            //cphvb_intp bundle_size = cphvb_inst_bundle(cur_bundle, 0, regular_size-1);
            cphvb_intp bundle_size = 1;
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

cphvb_error cphvb_ve_mcore_shutdown( void )
{
    //Shutdown the dispatcher.
    dispatch_finalize();
    return CPHVB_SUCCESS;

}

cphvb_error cphvb_ve_mcore_reg_func(char *fun, cphvb_intp *id) {

    if(strcmp("cphvb_reduce", fun) == 0)
    {
    	if (reduce_impl == NULL)
    	{
			cphvb_component_get_func(myself, fun, &reduce_impl);
			if (reduce_impl == NULL)
				return CPHVB_USERFUNC_NOT_SUPPORTED;

			reduce_impl_id = *id;
			return CPHVB_SUCCESS;			
        }
        else
        {
        	*id = reduce_impl_id;
        	return CPHVB_SUCCESS;
        }
    }
    else if(strcmp("cphvb_random", fun) == 0)
    {
    	if (random_impl == NULL)
    	{
			cphvb_component_get_func(myself, fun, &random_impl);
			if (random_impl == NULL)
				return CPHVB_USERFUNC_NOT_SUPPORTED;

			random_impl_id = *id;
			return CPHVB_SUCCESS;			
        }
        else
        {
        	*id = random_impl_id;
        	return CPHVB_SUCCESS;
        }
    }

    return CPHVB_USERFUNC_NOT_SUPPORTED;
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
    err = dispatch_bundle(inst,1,1);
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
        err = dispatch_bundle(inst,1,1);
        if(err != CPHVB_SUCCESS)
            return err;
        tmp.start += step;
    }

    return CPHVB_SUCCESS;
}
