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
#include "cphvb_ve_simple.h"
#include <cphvb_vcache.h>

static cphvb_component *myself = NULL;
static cphvb_userfunc_impl reduce_impl = NULL;
static cphvb_intp reduce_impl_id = 0;
static cphvb_userfunc_impl random_impl = NULL;
static cphvb_intp random_impl_id = 0;
static cphvb_userfunc_impl matmul_impl = NULL;
static cphvb_intp matmul_impl_id = 0;
static cphvb_userfunc_impl nselect_impl = NULL;
static cphvb_intp nselect_impl_id = 0;

static cphvb_intp vcache_size   = 10;

cphvb_error cphvb_ve_simple_init(cphvb_component *self)
{
    myself = self;

    char *env = getenv("CPHVB_CORE_VCACHE_SIZE");     // Override block_size from environment-variable.
    if(env != NULL)
    {
        vcache_size = atoi(env);
    }
    if(vcache_size <= 0)                        // Verify it
    {
        fprintf(stderr, "CPHVB_CORE_VCACHE_SIZE (%ld) should be greater than zero!\n", (long int)vcache_size);
        return CPHVB_ERROR;
    }

    cphvb_vcache_init( vcache_size );
    return CPHVB_SUCCESS;
}

cphvb_error cphvb_ve_simple_execute( cphvb_intp instruction_count, cphvb_instruction* instruction_list )
{
    cphvb_intp count;
    cphvb_instruction* inst;
    cphvb_error res;

    for (count=0; count < instruction_count; count++) {

        inst = &instruction_list[count];
        if (inst->status == CPHVB_SUCCESS) {        // SKIP instruction
            continue;
        }

        res = cphvb_vcache_malloc( inst );          // Allocate memory for operands
        if ( res != CPHVB_SUCCESS ) {
            printf("Unhandled error returned by cphvb_vcache_malloc() called from cphvb_ve_simple_execute()\n");
            return res;
        }
                                                    
        switch (inst->opcode) {                     // Dispatch instruction

            case CPHVB_NONE:                        // NOOP.
            case CPHVB_DISCARD:
            case CPHVB_SYNC:
                inst->status = CPHVB_SUCCESS;
                break;
            case CPHVB_FREE:                        // Store data-pointer in malloc-cache
                inst->status = cphvb_vcache_free( inst );
                break;

            case CPHVB_USERFUNC:                    // External libraries

                if (inst->userfunc->id == reduce_impl_id) {

                    inst->status = reduce_impl(inst->userfunc, NULL);

                } else if(inst->userfunc->id == random_impl_id) {

                    inst->status = random_impl(inst->userfunc, NULL);

                } else if(inst->userfunc->id == matmul_impl_id) {

                    inst->status = matmul_impl(inst->userfunc, NULL);

                } else if(inst->userfunc->id == nselect_impl_id) {

                    inst->status = nselect_impl(inst->userfunc, NULL);

                } else {                            // Unsupported userfunc
                
                    inst->status = CPHVB_USERFUNC_NOT_SUPPORTED;

                }

                break;

            default:                            // Built-in operations
                inst->status = cphvb_compute_apply( inst );

        }

        if (inst->status != CPHVB_SUCCESS) {    // Instruction failed
            break;
        }

    }

    if (count == instruction_count) {
        return CPHVB_SUCCESS;
    } else {
        return CPHVB_PARTIAL_SUCCESS;
    }

}

cphvb_error cphvb_ve_simple_shutdown( void )
{
    // De-allocate the malloc-cache
    cphvb_vcache_clear();
    cphvb_vcache_delete();

    return CPHVB_SUCCESS;
}

cphvb_error cphvb_ve_simple_reg_func(char *fun, cphvb_intp *id) 
{
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
    else if(strcmp("cphvb_nselect", fun) == 0)
    {
        if (nselect_impl == NULL)
        {
            cphvb_component_get_func(myself, fun, &nselect_impl);
            if (nselect_impl == NULL)
                return CPHVB_USERFUNC_NOT_SUPPORTED;
            
            nselect_impl_id = *id;
            return CPHVB_SUCCESS;
        }
        else
        {
            *id = nselect_impl_id;
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
    return cphvb_compute_matmul( arg, ve_arg );
}

cphvb_error cphvb_nselect( cphvb_userfunc *arg, void* ve_arg)
{
    return cphvb_compute_nselect( arg, ve_arg );
}

