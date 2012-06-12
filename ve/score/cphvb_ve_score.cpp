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
static computeloop* cphvb_ve_score_compute_loops = NULL;

static cphvb_intp block_size = 1000;

cphvb_error cphvb_ve_score_init(cphvb_component *self)
{
    myself = self;                              // Assign config container.

    char *env = getenv("CPHVB_VE_BLOCKSIZE");   // Override block_size from environment-variable.
    if(env != NULL)
    {
        block_size = atoi(env);
    }
    if(block_size <= 0)                         // Verify it
    {
        fprintf(stderr, "CPHVB_VE_BLOCKSIZE (%ld) should be greater than zero!\n", block_size);
        return CPHVB_ERROR;
    }

    return CPHVB_SUCCESS;
}

inline cphvb_error block_execute( cphvb_instruction* instr, cphvb_intp start, cphvb_intp end) {

    cphvb_intp i, k;

    //Make sure we have enough space
    if ((end - start + 1) > cphvb_ve_score_buffersizes) 
    {
    	cphvb_intp mcount = (end - start + 1);

    	//We only work in multiples of 1000
    	if (mcount % 1000 != 0)
    		mcount = (mcount + 1000) - (mcount % 1000);

    	//Make sure we re-allocate on error
    	cphvb_ve_score_buffersizes = 0;
    	
    	if (cphvb_ve_score_compute_loops != NULL) {
    		free(cphvb_ve_score_compute_loops);
    		cphvb_ve_score_compute_loops = NULL;
    	}
    	
    	cphvb_ve_score_compute_loops = (computeloop*)malloc(sizeof(computeloop) * mcount);
    	
    	if (cphvb_ve_score_compute_loops == NULL)
    		return CPHVB_OUT_OF_MEMORY;
    	
    	cphvb_ve_score_buffersizes = mcount;
    }
    
    computeloop* compute_loops = cphvb_ve_score_compute_loops;    
    /*
    // Strategy One:
    // Consecutively execute instructions one by one applying compute-loop immediately.
    for(i=start; i <= end; i++)
    {
        cphvb_compute_apply( &instr[i] );
        instr[i].status = CPHVB_INST_DONE;
    }
    */

    /*
    // Strategy Two:
    // Seperating execution into: grabbing instructions, executing them and lastly setting status.
    for(i=start, k=0; i <= end; i++,k++)            // Get the compute-loops
    {
        compute_loops[k] = cphvb_compute_get( &instr[i] );
    }

    for(i=start, k=0; i <= end; i++, k++)           // Execute them
    {
        compute_loops[k]( &instr[i], 0, 0 );
    }

    for(i=start; i <= end; i++)                     // Set instruction status
    {
        instr[i].status = CPHVB_INST_DONE;
    }
    */

    /*
    // Strategy Two.Five:
    // This is strategy three but without actually using the calculated offsets.
    // This should definately isolate the performance problem.
    //
    cphvb_intp  nelements,
                trav_start=0,
                trav_end=-1;

    for(i=start, k=0; i <= end; i++,k++)            // Get the compute-loops
    {
        compute_loops[k] = cphvb_compute_get( &instr[i] );
    }
                                                    // Block-size split
    nelements = cphvb_nelements( instr[start].operand[0]->ndim, instr[start].operand[0]->shape );
    while(nelements>0)
    {
        nelements -= block_size;
        trav_start = trav_end +1;
        if (nelements > 0) {
            trav_end = trav_start+ block_size-1;
        } else {
            trav_end = trav_start+ block_size-1 +nelements;
        }

        for(i=start, k=0; i <= end; i++, k++)
        {
            // no actual execution here...
        }
    }

    for(i=start, k=0; i <= end; i++, k++)           // Execute them
    {
        compute_loops[k]( &instr[i], 0, 0 );
    }

    for(i=start; i <= end; i++)                     // Set instruction status
    {
        instr[i].status = CPHVB_INST_DONE;
    }
    */

    
    // Strategy Three:
    // Same as two but also slice into blocks.
    cphvb_intp  nelements,
    trav_start=0,
    trav_end=-1;

    for(i=start, k=0; i <= end; i++,k++)            // Get the compute-loops
    {
        compute_loops[k] = cphvb_compute_get( &instr[i] );
    }
                                                    // Execute them
    nelements = cphvb_nelements( instr[start].operand[0]->ndim, instr[start].operand[0]->shape );
    //printf("Blocking up %ld elements in blocks of max  %ld!\n", nelements, block_size);
    while(nelements>0)
    {
        nelements -= block_size;
        trav_start = trav_end +1;
        if (nelements > 0) {
            trav_end = trav_start+ block_size-1;
        } else {
            trav_end = trav_start+ block_size-1 +nelements;
        }

        for(i=start, k=0; i <= end; i++, k++)
        {
            //printf("Instr[%ld], ELEMENTS: %ld-->%ld. \n", i, trav_start, trav_end);
            //cphvb_pprint_instr( &instr[i] );
            compute_loops[k]( &instr[i], trav_start, trav_end );
        }
    }

    for(i=start; i <= end; i++)                     // Set instruction status
    {
        instr[i].status = CPHVB_INST_DONE;
    }

    /*
    // Strategy Five:
    // No block-size splits, instead do a pseudo-split.
    //

    cphvb_intp nelements = cphvb_nelements( instr[start].operand[0]->ndim, instr[start].operand[0]->shape );
    //cphvb_intp split = 15728640;
    cphvb_intp split = nelements-99;
    //cphvb_intp split = 1;
    for(i=start, k=0; i <= end; i++,k++)            // Get the compute-loops
    {
        compute_loops[k] = cphvb_compute_get( &instr[i] );
    }

    for(i=start, k=0; i <= end; i++, k++)           // Execute them
    {
        compute_loops[k]( &instr[i], 0, split );
    }

    for(i=start, k=0; i <= end; i++, k++)           // Execute them
    {
        compute_loops[k]( &instr[i], split+1, nelements );
    }

    for(i=start; i <= end; i++)                     // Set instruction status
    {
        instr[i].status = CPHVB_INST_DONE;
    }
    */

    //
    //
    //
    //  THE LESSON LEARNED: COORDINATED BASED OFFSET COMPUTATION SUCKS! It is much too expensive!
    //
    //
    //
    
    return CPHVB_SUCCESS;

}

cphvb_error cphvb_ve_score_execute( cphvb_intp instruction_count, cphvb_instruction* instruction_list )
{
    cphvb_intp cur_index, nops, i, j;
    cphvb_instruction *inst, *binst;
    cphvb_error ret = CPHVB_SUCCESS;

    cphvb_intp bin_start, bin_end, bin_size;
    cphvb_intp bundle_start, bundle_end, bundle_size;

    for(cur_index=0; cur_index < instruction_count; cur_index++)
    {
        inst = &instruction_list[cur_index];

        if(inst->status == CPHVB_INST_DONE)     // SKIP instruction
        {
            continue;
        }

        nops = cphvb_operands(inst->opcode);    // Allocate memory for operands
        for(i=0; i<nops; i++)
        {
            if (!cphvb_is_constant(inst->operand[i]))
            {
                if (cphvb_data_malloc(inst->operand[i]) != CPHVB_SUCCESS)
                {
                    return CPHVB_OUT_OF_MEMORY; // EXIT
                }
            }

        }

        switch(inst->opcode)                    // Dispatch instruction
        {
            case CPHVB_NONE:                    // NOOP.
            case CPHVB_DISCARD:
            case CPHVB_SYNC:
                inst->status = CPHVB_INST_DONE;
                break;

            case CPHVB_USERFUNC:                // External libraries

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
                else if(inst->userfunc->id == matmul_impl_id)
                {
                    ret = matmul_impl(inst->userfunc, NULL);
                    inst->status = (ret == CPHVB_SUCCESS) ? CPHVB_INST_DONE : CPHVB_INST_UNDONE;
                }
                else                            // Unsupported userfunc
                {
                    inst->status = CPHVB_INST_NOT_SUPPORTED;
                }

                break;

            default:                                            // Built-in operations
                                                                // -={[ BINNING ]}=-
                bin_start   = cur_index;                        // Count built-ins and their start/end indexes and allocate memory for them.
                bin_end     = cur_index;
                for(j=bin_start+1; j<instruction_count; j++)    
                {
                    binst = &instruction_list[j];               // EXIT: Stop if instruction is NOT built-in
                    if ((binst->opcode == CPHVB_NONE) || (binst->opcode == CPHVB_DISCARD) || (binst->opcode == CPHVB_SYNC) ||(binst->opcode == CPHVB_USERFUNC) ) {
                        break;
                    }

                    bin_end++;                                  // The "end" index

                    nops = cphvb_operands(binst->opcode);       // The memory part...
                    for(i=0; i<nops; i++)
                    {
                        if (!cphvb_is_constant(binst->operand[i]))
                        {
                            if (cphvb_data_malloc(binst->operand[i]) != CPHVB_SUCCESS)
                            {
                                return CPHVB_OUT_OF_MEMORY;     // EXIT
                            }
                        }

                    }

                }
                bin_size = bin_end - bin_start +1;              // The counting part

                                                                // -={[ BUNDLING ]}=-
                bundle_size     = (bin_size > 1) ? cphvb_inst_bundle( instruction_list, bin_start, bin_end ) : 1;
                bundle_start    = bin_start;
                bundle_end      = bundle_start + bundle_size-1;

				//printf("\nINSTRUCTIONS: %ld-->%ld\n", bundle_start, bundle_end);
                block_execute( instruction_list, bundle_start, bundle_end );
                cur_index += bundle_size-1;
        }

        if (inst->status != CPHVB_INST_DONE)    // Instruction failed
        {
            break;
        }

    }

    if (cur_index == instruction_count) {
        return CPHVB_SUCCESS;
    } else {
        return CPHVB_PARTIAL_SUCCESS;
    }

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

