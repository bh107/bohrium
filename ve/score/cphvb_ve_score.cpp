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

static cphvb_component *myself = NULL;
static cphvb_userfunc_impl reduce_impl = NULL;
static cphvb_intp reduce_impl_id = 0;
static cphvb_userfunc_impl random_impl = NULL;
static cphvb_intp random_impl_id = 0;
static cphvb_userfunc_impl matmul_impl = NULL;
static cphvb_intp matmul_impl_id = 0;

static cphvb_intp cphvb_ve_score_buffersizes = 0;
static cphvb_instruction** cphvb_ve_score_instruction_list = NULL;
static computeloop* cphvb_ve_score_compute_loops = NULL;

cphvb_error cphvb_ve_score_init(cphvb_component *self)
{
    myself = self;
    return CPHVB_SUCCESS;
}

inline cphvb_error execute_range( cphvb_instruction* list, cphvb_intp start, cphvb_intp count) {

    cphvb_intp i, bundle_size, nelements,
                block_size=100, trav_start=0, trav_end=0, trav_len=100;
        
    //Make sure we have enough space
    if (count > cphvb_ve_score_buffersizes) 
    {
    	cphvb_intp mcount = count;

    	//We only work in multiples of 1000
    	if (mcount % 1000 != 0)
    		mcount = (mcount + 1000) - (mcount % 1000);
    		
    	//Make sure we re-allocate on error
    	cphvb_ve_score_buffersizes = 0;
    	
    	if (cphvb_ve_score_instruction_list != NULL) {
    		free(cphvb_ve_score_instruction_list);
    		cphvb_ve_score_instruction_list = NULL;
    	}

    	if (cphvb_ve_score_compute_loops != NULL) {
    		free(cphvb_ve_score_compute_loops);
    		cphvb_ve_score_compute_loops = NULL;
    	}
    	
    	cphvb_ve_score_instruction_list = (cphvb_instruction**)malloc(sizeof(cphvb_instruction*) * mcount);
    	cphvb_ve_score_compute_loops = (computeloop*)malloc(sizeof(computeloop) * mcount);
    	
    	if (cphvb_ve_score_instruction_list == NULL || cphvb_ve_score_compute_loops == NULL)
    		return CPHVB_OUT_OF_MEMORY;
    	
    	cphvb_ve_score_buffersizes = mcount;
    }
    
    cphvb_instruction** inst = cphvb_ve_score_instruction_list;
    computeloop* compute_loops = cphvb_ve_score_compute_loops;

    if (count > 1) {

        //cphvb_pprint_instr_list( &list[start], count, "POTENTIAL" );
        for(i=0; i < count; i++)
        {
            inst[i] = &list[start+i];
        }
        bundle_size = cphvb_inst_bundle( inst, count );

        for(i=0; i < bundle_size; i++)
        {
            compute_loops[i] = cphvb_compute_get( inst[i] );
        }

        //cphvb_pprint_instr_list( *inst, bundle_size, "BUNDLE" );

        /* Regular invocation of instructions.
        for(i=0; i<bundle_size; i++)
        {
            cphvb_compute_apply( inst[i] );
            inst[i]->status = CPHVB_INST_DONE;
        }*/

        nelements = cphvb_nelements( inst[0]->operand[0]->ndim, inst[0]->operand[0]->shape );
        while(nelements>0)
        {
            nelements -= block_size;
            if (nelements < 0) {
                trav_len = block_size + nelements;
            }
            trav_start   = trav_end;
            trav_end     = trav_start+trav_len;

            for(i=0; i < bundle_size; i++)      
            {
                compute_loops[i]( inst[i], trav_start, trav_end );
            }
        }

        for(i=0; i < bundle_size; i++)          // Set instruction status
        {
            inst[i]->status = CPHVB_INST_DONE;
        }

        for(i=bundle_size; i < count; i++)
        {
            cphvb_compute_apply( inst[i] );
            inst[i]->status = CPHVB_INST_DONE;
        }

    } else {

        for(i=start; i < (start+count); i++)
        {
            cphvb_compute_apply( &list[i] );
            list[i].status = CPHVB_INST_DONE;
        }

    }

    return CPHVB_SUCCESS;

}

cphvb_error cphvb_ve_score_execute( cphvb_intp instruction_count, cphvb_instruction* instruction_list )
{
    cphvb_intp count, nops, i, kernel_size;
    cphvb_instruction* inst;
    cphvb_error ret = CPHVB_SUCCESS;

    kernel_size = 0;
    for(count=0; count < instruction_count; count++)
    {
        inst = &instruction_list[count];

        if(inst->status == CPHVB_INST_DONE)         // SKIP instruction
        {
            execute_range( instruction_list, count-kernel_size, kernel_size );
            kernel_size = 0;

            continue;
        }

        nops = cphvb_operands(inst->opcode);        // Allocate memory for operands
        for(i=0; i<nops; i++)
        {
            if (!cphvb_is_constant(inst->operand[i]))
            {
                if (cphvb_data_malloc(inst->operand[i]) != CPHVB_SUCCESS)
                {
                    return CPHVB_OUT_OF_MEMORY;     // EXIT
                }
            }

        }

        switch(inst->opcode)                        // Dispatch instruction
        {
            case CPHVB_NONE:                        // NOOP.
            case CPHVB_DISCARD:
            case CPHVB_SYNC:
                execute_range( instruction_list, count-kernel_size, kernel_size );
                kernel_size = 0;

                inst->status = CPHVB_INST_DONE;
                break;

            case CPHVB_USERFUNC:                    // External libraries are executed

                execute_range( instruction_list, count-kernel_size, kernel_size );
                kernel_size = 0;

                if(inst->userfunc->id == reduce_impl_id)
                {
                    ret = reduce_impl(inst->userfunc, NULL);
                    inst->status = (ret == CPHVB_SUCCESS) ? CPHVB_INST_DONE : CPHVB_INST_UNDONE;
                }
                else if(inst->userfunc->id == random_impl_id)
                {
                    ret = random_impl(inst->userfunc, NULL);
                    inst->status = (ret == CPHVB_SUCCESS) ? CPHVB_INST_DONE : CPHVB_INST_UNDONE;
                }
                else                                // Unsupported userfunc
                {
                    ret = CPHVB_TYPE_NOT_SUPPORTED;
                }

                break;

            default:                                // Built-in operations
                kernel_size++;
        }
        
    }
                                                // All instructions succeeded.
    execute_range( instruction_list, count-kernel_size, kernel_size );
    kernel_size = 0;

    return ret;                                 // EXIT

}

cphvb_error cphvb_ve_score_shutdown( void )
{
    return CPHVB_SUCCESS;
}

cphvb_error cphvb_ve_score_reg_func(char *fun, cphvb_intp *id) {

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
    else if(strcmp("cphvb_matmul", fun) == 0)
    {
    	if (matmul_impl == NULL)
    	{
			cphvb_component_get_func(myself, fun, &matmul_impl);
			if (matmul_impl == NULL)
				return CPHVB_USERFUNC_NOT_SUPPORTED;

			matmul_impl_id = *id;
			return CPHVB_SUCCESS;			
        }
        else
        {
        	*id = matmul_impl_id;
        	return CPHVB_SUCCESS;
        }
    }
    
    return CPHVB_USERFUNC_NOT_SUPPORTED;
}

cphvb_error cphvb_reduce( cphvb_userfunc *arg, void* ve_arg)
{
    return cphvb_compute_reduce( arg, ve_arg );
}

cphvb_error cphvb_random( cphvb_userfunc *arg, void* ve_arg)
{
    return cphvb_compute_random( arg, ve_arg );
}

cphvb_error cphvb_matmul( cphvb_userfunc *arg, void* ve_arg)
{
    printf("SCORE doing Matrix Multiplication!\n");
    return CPHVB_SUCCESS;
}

