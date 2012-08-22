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
#include <cphvb_mcache.h>
//
// C-Friendly version of the mcache.
// 
static cphvb_data_ptr   *cphvb_mcache;
static cphvb_intp       *cphvb_mcache_bytes;
static cphvb_intp   cphvb_mcache_bytes_total;
static int          cphvb_mcache_size;
static int          cphvb_mcache_cur;

/**
 *  Initiate mcache to a fixed size.
 *
 *  cache will hold 'size' elements in the cache.
 *
 *  Expiration / cache invalidation is based on round-robin;
 *  Whenever an element is added to mcache the round-robin
 *  counter is incremented.
 *
 */
void cphvb_mcache_init( int size )
{
    cphvb_mcache_size           = size;
    cphvb_mcache_bytes_total    = 0;

    cphvb_mcache_cur    = 0;
    cphvb_mcache        = (cphvb_data_ptr*)malloc(sizeof(cphvb_data_ptr)*size);
    memset(cphvb_mcache,0,sizeof(cphvb_data_ptr)*size);
    cphvb_mcache_bytes  = (cphvb_intp*)malloc(sizeof(cphvb_intp)*size);
    memset(cphvb_mcache_bytes,0,sizeof(cphvb_intp)*size);
}

/**
 * Deallocates mcache arrays.
 */
void cphvb_mcache_delete()
{
    free( cphvb_mcache );
    free( cphvb_mcache_bytes );
}

/**
 * Remove all entries from mcache and de-allocate them
 */
void cphvb_mcache_clear()
{
    int i;
    for (i=0; i<cphvb_mcache_size; i++) {
        if (cphvb_mcache_bytes[i] > 0) {
            cphvb_memory_free( cphvb_mcache[i], cphvb_mcache_bytes[i] );
        }
    }
}

/**
 * Return and remove a pointer of size 'size' from mcache.
 *
 * This removes it from the mcache!
 *
 * @return null If none exists.
 */
cphvb_data_ptr cphvb_mcache_find( cphvb_intp bytes )
{
    int i;
    for (i=0; i<cphvb_mcache_size; i++) {
        if (cphvb_mcache_bytes[i] == bytes) {
            DEBUG_PRINT("Found %p in mcache.\n", cphvb_mcache[i]);
            cphvb_mcache_bytes[i] = 0;
            cphvb_mcache_bytes_total -= bytes;
            return cphvb_mcache[i];
        }
    }
    return NULL;
}

/**
 * Add an element to mcache.
 * 
 * @param data Pointer to allocated data.
 * @param size Size in bytes of the allocated data.
 */
void cphvb_mcache_insert( cphvb_data_ptr data, cphvb_intp size )
{
    if (cphvb_mcache_bytes[cphvb_mcache_cur] > 0) {
		DEBUG_PRINT("Free=%p\n", cphvb_mcache[cphvb_mcache_cur]);
        cphvb_memory_free( cphvb_mcache[cphvb_mcache_cur], cphvb_mcache_bytes[cphvb_mcache_cur] );
    }

    cphvb_mcache[cphvb_mcache_cur] = data;
    cphvb_mcache_bytes[cphvb_mcache_cur] = size;

    cphvb_mcache_cur = (cphvb_mcache_cur+1) % cphvb_mcache_size;
    cphvb_mcache_bytes_total += size;
}

cphvb_error cphvb_mcache_free( cphvb_instruction* inst )
{
    cphvb_array* base;
    cphvb_intp nelements, bytes;

    base = cphvb_base_array( inst->operand[0] ); 
    if (base->data != NULL) {
        nelements   = cphvb_nelements(base->ndim, base->shape);
        bytes       = nelements * cphvb_type_size(base->type);
        
		DEBUG_PRINT("Deallocate=%p\n", base->data);
        cphvb_mcache_insert( base->data, bytes );
		base->data = NULL;
    }
    inst->operand[0] = NULL;

    return CPHVB_SUCCESS;
}

cphvb_error cphvb_mcache_malloc( cphvb_instruction* inst )
{
    cphvb_array* base;
    cphvb_intp nops, i, nelements, bytes;

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
                            base->data = cphvb_mcache_find( bytes );
                            if (base->data == NULL) {
                                if (cphvb_data_malloc(inst->operand[0]) != CPHVB_SUCCESS) {
                                    inst->status = CPHVB_OUT_OF_MEMORY;
                                    return CPHVB_OUT_OF_MEMORY;         // EXIT
                                }                                   
                                DEBUG_PRINT("Allocated=%p\n", base->data);
                            } else {
                                DEBUG_PRINT("Reusing=%p.\n", base->data);
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

    return CPHVB_SUCCESS;

}
