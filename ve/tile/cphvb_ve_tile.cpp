/*
This file is part of cphVB and copyright (c) 2012 the cphVB team:
http://cphvb.bitbucket.org

cphVB is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as 
published by the Free Software Foundation, either version 3 
of the License, or (at your option) any later version.

cphVB is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the 
GNU Lesser General Public License along with cphVB. 

If not, see <http://www.gnu.org/licenses/>.
*/

#include <cphvb.h>
#include "cphvb_ve_tile.h"
#include <iostream>
#include <cphvb_vcache.h>

static cphvb_component *myself = NULL;
static cphvb_userfunc_impl reduce_impl = NULL;
static cphvb_intp reduce_impl_id = 0;
static cphvb_userfunc_impl random_impl = NULL;
static cphvb_intp random_impl_id = 0;
static cphvb_userfunc_impl matmul_impl = NULL;
static cphvb_intp matmul_impl_id = 0;
static cphvb_userfunc_impl lu_impl = NULL;
static cphvb_intp lu_impl_id = 0;
static cphvb_userfunc_impl fft_impl = NULL;
static cphvb_intp fft_impl_id = 0;
static cphvb_userfunc_impl fft2_impl = NULL;
static cphvb_intp fft2_impl_id = 0;

static cphvb_intp cphvb_ve_tile_buffersizes = 0;
static cphvb_computeloop_naive* cphvb_ve_tile_compute_loops = NULL;
static cphvb_tstate_naive* cphvb_ve_tile_tstates = NULL;

static cphvb_intp block_size    = 7000;
static cphvb_intp bin_max       = 20;
static cphvb_intp base_max      = 7;
static cphvb_intp vcache_size   = 10;

cphvb_error cphvb_ve_tile_init(cphvb_component *self)
{
    myself = self;                              // Assign config container.

    char *env = getenv("CPHVB_VE_TILE_BLOCKSIZE");   // Override block_size from environment-variable.
    if(env != NULL)
    {
        block_size = atoi(env);
    }
    if(block_size <= 0)                         // Verify it
    {
        fprintf(stderr, "CPHVB_VE_TILE_BLOCKSIZE (%ld) should be greater than zero!\n", (long int)block_size);
        return CPHVB_ERROR;
    }

    env = getenv("CPHVB_VE_TILE_BINMAX");      // Override block_size from environment-variable.
    if(env != NULL)
    {
        bin_max = atoi(env);
    }
    if(bin_max <= 0)                            // Verify it
    {
        fprintf(stderr, "CPHVB_VE_TILE_BINMAX (%ld) should be greater than zero!\n", (long int)bin_max);
        return CPHVB_ERROR;
    }

    env = getenv("CPHVB_VE_TILE_BASEMAX");      // Override base max from environment-variable.
    if(env != NULL)
    {
        base_max = atoi(env);
    }
    if(base_max <= 0)                            // Verify it
    {
        fprintf(stderr, "CPHVB_VE_TILE_BASEMAX (%ld) should be greater than zero!\n", (long int)base_max);
        return CPHVB_ERROR;
    }

    env = getenv("CPHVB_CORE_VCACHE_SIZE");     // Override block_size from environment-variable.
    if(env != NULL)
    {
        vcache_size = atoi(env);
    }
    if(vcache_size <= 0)                        // Verify it
    {
        fprintf(stderr, "CPHVB_CORE_VCACHE_SIZE (%ld) should be greater than zero!\n", (long int)vcache_size);
        return CPHVB_ERROR;
    }
    //printf("[CPHVB_VE_TILE_BLOCKSIZE=%ld]\n", block_size);
    //printf("[CPHVB_VE_TILE_BINMAX=%ld]\n", bin_max);
    //printf("[CPHVB_VE_TILE_BASEMAX=%ld]\n", base_max);
    //printf("[CPHVB_CORE_VCACHE_SIZE=%ld]\n", vcache_size);
    cphvb_vcache_init( vcache_size );
    return CPHVB_SUCCESS;
}

inline cphvb_error block_execute( cphvb_instruction* instr, cphvb_intp start, cphvb_intp end) {

    cphvb_intp i, k;

    if ((end - start + 1) > cphvb_ve_tile_buffersizes) // Make sure we have enough space
    {
    	cphvb_intp mcount = (end - start + 1);

    	if (mcount % 1000 != 0)                         // We only work in multiples of 1000
    		mcount = (mcount + 1000) - (mcount % 1000);

    	cphvb_ve_tile_buffersizes = 0;                 // Make sure we re-allocate on error
    	
    	if (cphvb_ve_tile_compute_loops != NULL) {
    		free(cphvb_ve_tile_compute_loops);
    		free(cphvb_ve_tile_tstates);
    		cphvb_ve_tile_compute_loops = NULL;
    		cphvb_ve_tile_tstates = NULL;
    	}
    	
    	cphvb_ve_tile_compute_loops    = (cphvb_computeloop_naive*)malloc(sizeof(cphvb_computeloop_naive) * mcount);
        cphvb_ve_tile_tstates          = (cphvb_tstate_naive*)malloc(sizeof(cphvb_tstate_naive)*mcount);
    	
    	if (cphvb_ve_tile_compute_loops == NULL)
    		return CPHVB_OUT_OF_MEMORY;
    	if (cphvb_ve_tile_tstates == NULL)
    		return CPHVB_OUT_OF_MEMORY;
    	
    	cphvb_ve_tile_buffersizes = mcount;
    }
    
    cphvb_computeloop_naive* compute_loops = cphvb_ve_tile_compute_loops;
    cphvb_tstate_naive* states = cphvb_ve_tile_tstates;
    cphvb_intp  nelements, trav_end=0;
    cphvb_error ret_errcode = CPHVB_SUCCESS;
    
    for(i=0; i<= (end-start);i++)                   // Reset traversal coordinates
        cphvb_tstate_reset_naive( &states[i] );

    for(i=start, k=0; i <= end; i++,k++)            // Get the compute-loops
    {
        switch(instr[i].opcode) {                   // Ignore sys-ops
            case CPHVB_DISCARD:
            case CPHVB_FREE:
            case CPHVB_SYNC:
            case CPHVB_NONE:
            case CPHVB_USERFUNC:
                break;

            default:
                compute_loops[k] = cphvb_compute_get_naive( &instr[i] );
                if(compute_loops[k] == NULL)
                {
                    end             = start + k - 1;
                    instr[i].status = CPHVB_TYPE_NOT_SUPPORTED;
                    ret_errcode     = CPHVB_PARTIAL_SUCCESS;
                    break;
                }
                else
                    instr[i].status = CPHVB_SUCCESS;
        }

    }
                                                    // Block-size split
    nelements = cphvb_nelements( instr[start].operand[0]->ndim, instr[start].operand[0]->shape );
    while(nelements>trav_end)
    {
        trav_end    += block_size;
        if (trav_end > nelements) {
            trav_end = nelements;
        }
        
        for(i=start, k=0; i <= end; i++, k++)
        {
            switch(instr[i].opcode) {               // Ignore sys-ops
                case CPHVB_DISCARD:
                case CPHVB_FREE:
                case CPHVB_SYNC:
                case CPHVB_NONE:
                case CPHVB_USERFUNC:
                    break;

                default:
                    compute_loops[k]( &instr[i], &states[k], trav_end );
            }
        }
    }
    
    return ret_errcode;
}

cphvb_error cphvb_ve_tile_execute( cphvb_intp instruction_count, cphvb_instruction* instruction_list )
{
    cphvb_intp cur_index,  j;
    cphvb_instruction *inst, *binst;

    cphvb_intp bin_start, bin_end, bin_size;
    cphvb_intp bundle_start, bundle_end, bundle_size;
    cphvb_error res;

    for(cur_index=0; cur_index < instruction_count; cur_index++)
    {
        inst = &instruction_list[cur_index];
    }

    for(cur_index=0; cur_index < instruction_count; cur_index++)
    {
        inst = &instruction_list[cur_index];

        if(inst->status == CPHVB_SUCCESS)       // SKIP instruction
        {
            continue;
        }

        res = cphvb_vcache_malloc( inst );      // Allocate memory for operands
        if ( res != CPHVB_SUCCESS ) {
            return res;
        }
        
        switch(inst->opcode)                    // Dispatch instruction
        {
            case CPHVB_NONE:                    // NOOP.
            case CPHVB_DISCARD:
            case CPHVB_SYNC:
                inst->status = CPHVB_SUCCESS;
                break;

            case CPHVB_FREE:                        // Store data-pointer in malloc-cache
                inst->status = cphvb_vcache_free( inst );
                break;

            case CPHVB_USERFUNC:                // External libraries

                if(inst->userfunc->id == reduce_impl_id)
                {
                    inst->status = reduce_impl(inst->userfunc, NULL);
                }
                else if(inst->userfunc->id == random_impl_id)
                {
                    inst->status = random_impl(inst->userfunc, NULL);
                }
                else if(inst->userfunc->id == matmul_impl_id)
                {
                    inst->status = matmul_impl(inst->userfunc, NULL);
                }
                else if(inst->userfunc->id == lu_impl_id)
                {
                    inst->status = lu_impl(inst->userfunc, NULL);
                }
                else if(inst->userfunc->id == fft_impl_id)
                {
                    inst->status = fft_impl(inst->userfunc, NULL);
                }
                else if(inst->userfunc->id == fft2_impl_id)
                {
                    inst->status = fft2_impl(inst->userfunc, NULL);
                }
                else                            // Unsupported userfunc
                {
                    inst->status = CPHVB_USERFUNC_NOT_SUPPORTED;
                }

                break;

            default:                                            // Built-in operations
                                                                // -={[ BINNING ]}=-
                bin_start   = cur_index;                        // Count built-ins and their start/end indexes and allocate memory for them.
                bin_end     = cur_index;
                bool has_sys = false;
                for(j=bin_start+1; (j<instruction_count) && (j<bin_start+1+bin_max); j++)
                {
                    binst = &instruction_list[j];               
                    
                    if (binst->opcode == CPHVB_USERFUNC) {      // Stop bundle, userfunc encountered.
                        break;
                    }
                                                                // Delay sys-op
                    if ((binst->opcode == CPHVB_NONE) || \
                        (binst->opcode == CPHVB_DISCARD) || \
                        (binst->opcode == CPHVB_SYNC) || \
                        (binst->opcode == CPHVB_FREE) ) {
                        has_sys = true;
                    }

                    bin_end++;                                  // The "end" index
                    res = cphvb_vcache_malloc( binst );         // Allocate memory for operands
                    if ( res != CPHVB_SUCCESS ) {
                        return res;
                    }
                }
                bin_size = bin_end - bin_start +1;              // The counting part

                                                                // -={[ BUNDLING ]}=-
                bundle_size     = (bin_size > 1) ? cphvb_inst_bundle( instruction_list, bin_start, bin_end, base_max ) : 1;
                bundle_start    = bin_start;
                bundle_end      = bundle_start + bundle_size-1;

                //printf("Bundle: %ld %ld -> %ld\n", bundle_size, bundle_start, bundle_end);
                if (bundle_size > 1) { 

                    block_execute( instruction_list, bundle_start, bundle_end );
                    if (has_sys) {
                        for(j=bundle_start; j <= bundle_end; j++) {
                            inst = &instruction_list[j];
                            switch(inst->opcode)                    // Dispatch instruction
                            {
                                case CPHVB_NONE:                    // NOOP.
                                case CPHVB_DISCARD:
                                case CPHVB_SYNC:
                                    inst->status = CPHVB_SUCCESS;
                                    break;

                                case CPHVB_FREE:                        // Store data-pointer in malloc-cache
                                    inst->status = cphvb_vcache_free( inst );
                                    break;

                                       
                            }
                        }
                    }

                } else {
                    inst = &instruction_list[bundle_start];
                    inst->status = cphvb_compute_apply_naive( inst );
                }
                cur_index += bundle_size-1;

        }

        if (inst->status != CPHVB_SUCCESS)    // Instruction failed
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

cphvb_error cphvb_ve_tile_shutdown( void )
{
    cphvb_vcache_clear();
    cphvb_vcache_delete();

    return CPHVB_SUCCESS;
}

cphvb_error cphvb_ve_tile_reg_func(char *fun, cphvb_intp *id) {

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
    else if(strcmp("cphvb_lu", fun) == 0)
    {
    	if (lu_impl == NULL)
    	{
			cphvb_component_get_func(myself, fun, &lu_impl);
			if (lu_impl == NULL)
				return CPHVB_USERFUNC_NOT_SUPPORTED;

			lu_impl_id = *id;
			return CPHVB_SUCCESS;			
        }
        else
        {
        	*id = lu_impl_id;
        	return CPHVB_SUCCESS;
        }
    }
    else if(strcmp("cphvb_fft", fun) == 0)
    {
    	if (fft_impl == NULL)
    	{
			cphvb_component_get_func(myself, fun, &fft_impl);
			if (fft_impl == NULL)
				return CPHVB_USERFUNC_NOT_SUPPORTED;

			fft_impl_id = *id;
			return CPHVB_SUCCESS;			
        }
        else
        {
        	*id = fft_impl_id;
        	return CPHVB_SUCCESS;
        }
    }
    else if(strcmp("cphvb_fft2", fun) == 0)
    {
    	if (fft2_impl == NULL)
    	{
			cphvb_component_get_func(myself, fun, &fft2_impl);
			if (fft2_impl == NULL)
				return CPHVB_USERFUNC_NOT_SUPPORTED;

			fft2_impl_id = *id;
			return CPHVB_SUCCESS;			
        }
        else
        {
        	*id = fft2_impl_id;
        	return CPHVB_SUCCESS;
        }
    }
    
    return CPHVB_USERFUNC_NOT_SUPPORTED;
}

cphvb_error cphvb_reduce( cphvb_userfunc *arg, void* ve_arg)
{
    return cphvb_compute_reduce_naive( arg, ve_arg );
}

cphvb_error cphvb_random( cphvb_userfunc *arg, void* ve_arg)
{
    return cphvb_compute_random( arg, ve_arg );
}

cphvb_error cphvb_matmul( cphvb_userfunc *arg, void* ve_arg)
{
    return cphvb_compute_matmul( arg, ve_arg );
    
}
