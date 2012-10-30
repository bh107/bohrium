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
#include <cphvb_vcache.h>
//
// C-Friendly version of the vcache.
// 
static cphvb_data_ptr   *cphvb_vcache;
static cphvb_intp       *cphvb_vcache_bytes;
static cphvb_intp   cphvb_vcache_bytes_total;
static int          cphvb_vcache_size;
static int          cphvb_vcache_cur;

static int cphvb_vcache_hits = 0;
static int cphvb_vcache_miss = 0;
static int cphvb_vcache_store = 0;
static int cphvb_vcache_flush = 0;

/**
 *  Reset vcache counters
 */
inline
void cphvb_vcache_reset_counters() {
    cphvb_vcache_hits = 0;
    cphvb_vcache_miss = 0;
    cphvb_vcache_store = 0;
    cphvb_vcache_flush = 0;
}

/**
 *  Initiate vcache to a fixed size.
 *
 *  cache will hold 'size' elements in the cache.
 *
 *  Expiration / cache invalidation is based on round-robin;
 *  Whenever an element is added to vcache the round-robin
 *  counter is incremented.
 *
 */
void cphvb_vcache_init( int size )
{
    cphvb_vcache_size           = size;
    cphvb_vcache_bytes_total    = 0;

    cphvb_vcache_cur    = 0;
    cphvb_vcache        = (cphvb_data_ptr*)malloc(sizeof(cphvb_data_ptr)*size);
    memset(cphvb_vcache,0,sizeof(cphvb_data_ptr)*size);
    cphvb_vcache_bytes  = (cphvb_intp*)malloc(sizeof(cphvb_intp)*size);
    memset(cphvb_vcache_bytes,0,sizeof(cphvb_intp)*size);

    cphvb_vcache_reset_counters();
}

/**
 * Deallocates vcache arrays.
 */
void cphvb_vcache_delete()
{
    free( cphvb_vcache );
    free( cphvb_vcache_bytes );
}

/**
 * Remove all entries from vcache and de-allocate them
 */
void cphvb_vcache_clear()
{
    int i;
    for (i=0; i<cphvb_vcache_size; i++) {
        if (cphvb_vcache_bytes[i] > 0) {
            cphvb_memory_free( cphvb_vcache[i], cphvb_vcache_bytes[i] );
        }
    }
}

/**
 * Return and remove a pointer of size 'size' from vcache.
 *
 * This removes it from the vcache!
 *
 * @return null If none exists.
 */
cphvb_data_ptr cphvb_vcache_find( cphvb_intp bytes )
{
    int i;
    for (i=0; i<cphvb_vcache_size; i++) {
        if (cphvb_vcache_bytes[i] == bytes) {
            DEBUG_PRINT("Found %p in vcache.\n", cphvb_vcache[i]);
            cphvb_vcache_hits++;
            cphvb_vcache_bytes[i] = 0;
            cphvb_vcache_bytes_total -= bytes;
            return cphvb_vcache[i];
        }
    }
    cphvb_vcache_miss++;
    return NULL;
}

/**
 * Add an element to vcache.
 * 
 * @param data Pointer to allocated data.
 * @param size Size in bytes of the allocated data.
 */
void cphvb_vcache_insert( cphvb_data_ptr data, cphvb_intp size )
{
    if (cphvb_vcache_bytes[cphvb_vcache_cur] > 0) {
		DEBUG_PRINT("Free=%p\n", cphvb_vcache[cphvb_vcache_cur]);
        cphvb_memory_free( cphvb_vcache[cphvb_vcache_cur], cphvb_vcache_bytes[cphvb_vcache_cur] );
    }

    cphvb_vcache[cphvb_vcache_cur] = data;
    cphvb_vcache_bytes[cphvb_vcache_cur] = size;

    cphvb_vcache_cur = (cphvb_vcache_cur+1) % cphvb_vcache_size;
    cphvb_vcache_bytes_total += size;

    cphvb_vcache_store++;
}

/**
 * De-allocate memory for an instructions output operand.
 *
 */
cphvb_error cphvb_vcache_free( cphvb_instruction* inst )
{
    cphvb_array* base;
    cphvb_intp nelements, bytes;

    base = cphvb_base_array( inst->operand[0] ); 
    if (base->data != NULL) {
        nelements   = cphvb_nelements(base->ndim, base->shape);
        bytes       = nelements * cphvb_type_size(base->type);
        
		DEBUG_PRINT("Deallocate=%p\n", base->data);
        cphvb_vcache_insert( base->data, bytes );
        cphvb_vcache_store++;
		base->data = NULL;
    }
    inst->operand[0] = NULL;

    return CPHVB_SUCCESS;
}

/**
 * Allocate memory for the output operand of the given instruction.
 * A previous (now unused) memory allocation will be assigned if available.
 *
 */
cphvb_error cphvb_vcache_malloc( cphvb_instruction* inst )
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
                            base->data = cphvb_vcache_find( bytes );
                            if (base->data == NULL) {
                                if (cphvb_data_malloc(inst->operand[0]) != CPHVB_SUCCESS) {
                                    inst->status = CPHVB_OUT_OF_MEMORY;
                                    return CPHVB_OUT_OF_MEMORY;         // EXIT
                                }                                   
                                DEBUG_PRINT("Allocated=%p\n", base->data);
                                cphvb_vcache_miss++;
                            } else {
                                DEBUG_PRINT("Reusing=%p.\n", base->data);
                                cphvb_vcache_hits++;
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
