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
#include "cphvb_ve_simple.h"
#include <map>

static cphvb_component *myself = NULL;
static cphvb_userfunc_impl reduce_impl = NULL;
static cphvb_intp reduce_impl_id = 0;
static cphvb_userfunc_impl random_impl = NULL;
static cphvb_intp random_impl_id = 0;
static cphvb_userfunc_impl matmul_impl = NULL;
static cphvb_intp matmul_impl_id = 0;

static std::multimap<cphvb_intp, cphvb_data_ptr> cache;                 // Malloc-cache
static std::multimap<cphvb_intp, cphvb_data_ptr>::iterator cache_it;
static cphvb_intp cachesize = 0;

cphvb_error cphvb_ve_simple_init(cphvb_component *self)
{
    myself = self;
    return CPHVB_SUCCESS;
}

cphvb_error cphvb_ve_simple_execute( cphvb_intp instruction_count, cphvb_instruction* instruction_list )
{
    cphvb_intp count, nops, i, nelements, bytes;
    cphvb_instruction* inst;
    cphvb_array* base;

    for (count=0; count < instruction_count; count++) {

        inst = &instruction_list[count];

        if (inst->status == CPHVB_SUCCESS) {        // SKIP instruction
            continue;
        }

        switch (inst->opcode) {                     // Allocate memory for built-in

            case CPHVB_NONE:                        // No memory operations for these
            case CPHVB_DISCARD:
            case CPHVB_SYNC:
            case CPHVB_USERFUNC:
            case CPHVB_FREE:
                break;

            default:                                    

                nops = cphvb_operands(inst->opcode);    // Allocate memory for operands        
                for(i=0; i<nops; i++)
                {
                    if (!cphvb_is_constant(inst->operand[i]))
                    {
                        //We allow allocation of the output operand only
                        if (i == 0)
                        {
                            base = cphvb_base_array( inst->operand[0] );    // Reuse or allocate
                            if (base->data == NULL) {

                                nelements   = cphvb_nelements(base->ndim, base->shape);
                                bytes       = nelements * cphvb_type_size(base->type);

                                DEBUG_PRINT("Reuse or allocate...\n");
                                cache_it = cache.find(bytes);
                                if (cache_it == cache.end()) {              // 1 - Allocate

                                    DEBUG_PRINT("Allocating.\n");
                                    if (cphvb_data_malloc(inst->operand[0]) != CPHVB_SUCCESS) {
                                        inst->status = CPHVB_OUT_OF_MEMORY;
                                        return CPHVB_OUT_OF_MEMORY;         // EXIT
                                    }

                                } else {                                    // 2 - Reuse

                                    DEBUG_PRINT("Reusing=%p.\n", cache_it->second);
                                    base->data = cache_it->second;
                                    cache.erase( cache_it );
                                    cachesize -= bytes;
                                }
                            }
                        }
                        else if(cphvb_base_array(inst->operand[i])->data == NULL) 
                        {
                            inst->status = CPHVB_ERROR;
                            return CPHVB_ERROR; // EXIT
                        }
                    }

                }
                break;

        }
                                                    
        switch (inst->opcode) {                     // Dispatch instruction

            case CPHVB_NONE:                        // NOOP.
            case CPHVB_DISCARD:
            case CPHVB_SYNC:
                inst->status = CPHVB_SUCCESS;
                break;
            case CPHVB_FREE:                        // Store data-pointer in malloc-cache

                base = cphvb_base_array( inst->operand[0] ); 
                if (base->data != NULL) {
                    nelements   = cphvb_nelements(base->ndim, base->shape);
                    bytes       = nelements * cphvb_type_size(base->type);

                    cache.insert(std::pair<cphvb_intp, cphvb_data_ptr>(bytes, base->data));
                    cachesize += bytes;
                }
                inst->operand[0] = NULL;

                inst->status = CPHVB_SUCCESS;
                break;

            case CPHVB_USERFUNC:                    // External libraries

                if (inst->userfunc->id == reduce_impl_id) {

                    inst->status = reduce_impl(inst->userfunc, NULL);

                } else if(inst->userfunc->id == random_impl_id) {

                    inst->status = random_impl(inst->userfunc, NULL);

                } else if(inst->userfunc->id == matmul_impl_id) {

                    inst->status = matmul_impl(inst->userfunc, NULL);

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
    cphvb_array* tmp;
    for (cache_it = cache.begin(); cache_it != cache.end(); cache_it++) {
        tmp = (cphvb_array*)cache_it->second;
        cphvb_memory_free( tmp, cache_it->first );
    }

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
