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

static cphvb_com *myself = NULL;
static cphvb_userfunc_impl reduce_impl = NULL;
static cphvb_intp reduce_impl_id = 0;
static cphvb_userfunc_impl random_impl = NULL;
static cphvb_intp random_impl_id = 0;
static cphvb_userfunc_impl matmul_impl = NULL;
static cphvb_intp matmul_impl_id = 0;

cphvb_error cphvb_ve_score_init(cphvb_com *self)
{
    myself = self;
    return CPHVB_SUCCESS;
}

inline cphvb_error block_execute( cphvb_instruction* instr, cphvb_intp start, cphvb_intp end) {

    cphvb_intp i, k, nelements,
                block_size=100,
                trav_start=0, trav_end=0, trav_len=100;
    computeloop compute_loops[CPHVB_MAX_NO_INST];

    /*
    // Strategy One:
    // Consecutively execute instructions one by one applying compute-loop immediately.
    for(i=start; i <= end; i++)
    {
        cphvb_compute_apply( &instr[i] );
        instr[i].status = CPHVB_INST_DONE;
    }
    */
    
    // Strategy Two:
    // Seperating execution into: grabbing instructions, executing them and lastly setting status.
    for(i=start, k=0; i <= end; i++,k++)            // Get the compute-loops
    {
        compute_loops[k] = cphvb_compute_get( &instr[i] );
    }

    for(i=start, k=0; i <= end; i++, k++)
    {
        compute_loops[k]( &instr[i], 0, 0 );
    }

    for(i=start; i <= end; i++)                     // Set instruction status
    {
        instr[i].status = CPHVB_INST_DONE;
    }
    /*
    // Strategy Three:
    // Same as two but also slice into blocks.

    for(i=start, k=0; i <= end; i++,k++)                // Get the compute-loops
    {
        compute_loops[k] = cphvb_compute_get( &instr[i] );
    }
                                            // Execute them
    nelements = cphvb_nelements( instr[0].operand[0]->ndim, instr[0].operand[0]->shape );
    while(nelements>0)
    {
        nelements -= block_size;
        if (nelements < 0) {
            trav_len = block_size + nelements;
        }
        trav_start   = trav_end;
        trav_end     = trav_start+trav_len;

        for(i=start, k=0; i <= end; i++, k++)
        {
            compute_loops[k]( &instr[i], trav_start, trav_end );
        }
    }

    for(i=start; i <= end; i++)          // Set instruction status
    {
        instr[i].status = CPHVB_INST_DONE;
    }
    */

    return CPHVB_SUCCESS;

}

cphvb_error cphvb_ve_score_execute( cphvb_intp instruction_count, cphvb_instruction* instruction_list )
{
    cphvb_intp count, nops, i, j;
    cphvb_instruction *inst, *binst;
    cphvb_error ret = CPHVB_SUCCESS;

    cphvb_intp krn_start, krn_end, krn_size;

    for(count=0; count < instruction_count; count++)
    {
        inst = &instruction_list[count];

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
                else                            // Unsupported userfunc
                {
                    inst->status = CPHVB_INST_UNDONE;
                }

                break;

            default:                            // Built-in operations

                krn_start   = count;
                krn_end     = count;

                for(j=krn_start+1; j<instruction_count; j++)    // Bundle built-in operations together
                {
                    binst = &instruction_list[j];
                    if ((binst->opcode == CPHVB_NONE) || (binst->opcode == CPHVB_DISCARD) || (binst->opcode == CPHVB_SYNC) ||(binst->opcode == CPHVB_USERFUNC) ) {
                        break;
                    }

                    krn_end++;                              // Extend bundle
                    nops = cphvb_operands(binst->opcode);   // Allocate memory for operands
                    for(i=0; i<nops; i++)
                    {
                        if (!cphvb_is_constant(binst->operand[i]))
                        {
                            if (cphvb_data_malloc(binst->operand[i]) != CPHVB_SUCCESS)
                            {
                                return CPHVB_OUT_OF_MEMORY; // EXIT
                            }
                        }

                    }

                }
   
                krn_size = krn_end - krn_start +1; 
                krn_size = (krn_size > 1) ? cphvb_inst_bundle( instruction_list, krn_start, krn_end ) : 1;
                krn_end = krn_start + krn_size-1;

                if (krn_size>1) {
                    //printf("%ld %ld %ld\n", krn_start, krn_end, krn_size);
                    block_execute( instruction_list, krn_start, krn_end );
                    count += krn_size-1;
                } else {
                    ret = cphvb_compute_apply( inst );
                    inst->status = (ret == CPHVB_SUCCESS) ? CPHVB_INST_DONE : CPHVB_INST_UNDONE;
                }
       }

        if (inst->status != CPHVB_INST_DONE)    // Instruction failed
        {
            break;
        }

    }

    if (count == instruction_count) {
        return CPHVB_SUCCESS;
    } else {
        return CPHVB_PARTIAL_SUCCESS;
    }

}

cphvb_error cphvb_ve_score_shutdown( void )
{
    return CPHVB_SUCCESS;
}

cphvb_error cphvb_ve_score_reg_func(char *lib, char *fun, cphvb_intp *id) {

    if(strcmp("cphvb_reduce", fun) == 0)
    {
    	if (reduce_impl == NULL)
    	{
			cphvb_com_get_func(myself, lib, fun, &reduce_impl);
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
			cphvb_com_get_func(myself, lib, fun, &random_impl);
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
			cphvb_com_get_func(myself, lib, fun, &matmul_impl);
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

