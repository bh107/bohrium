/*
This file is part of Bohrium and copyright (c) 2012 the Bohrium
team <http://www.bh107.org>.

Bohrium is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as 
published by the Free Software Foundation, either version 3 
of the License, or (at your option) any later version.

Bohrium is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the 
GNU Lesser General Public License along with Bohrium. 

If not, see <http://www.gnu.org/licenses/>.
*/
#include <errno.h>
#include <bh_vcache.h>
//
// C-Friendly version of the vcache.
// 
static bh_data_ptr   *bh_vcache;
static bh_intp       *bh_vcache_bytes;
static bh_intp  bh_vcache_bytes_total;
static int      bh_vcache_size;
static int      bh_vcache_cur;

static int bh_vcache_hits = 0;
static int bh_vcache_miss = 0;
static int bh_vcache_store = 0;
static int bh_vcache_flush = 0;

/**
 *  Reset vcache counters
 */
inline
void bh_vcache_reset_counters() {
    bh_vcache_hits = 0;
    bh_vcache_miss = 0;
    bh_vcache_store = 0;
    bh_vcache_flush = 0;
}

/**
 *  Initiate vcache to a fixed size.
 *
 *  cache will hold 'size' elements in the cache.
 *
 *  Expiration / cache invalidation is based on round-robin;
 *  Whenever an element is added to vcache the round-robin
 *  counter is incremented.
 */
void bh_vcache_init(int size)
{
    bh_vcache_size           = size;
    bh_vcache_bytes_total    = 0;
    bh_vcache_cur    = 0;

    if (0 < size) { // Enabled
        bh_vcache        = (bh_data_ptr*)malloc(sizeof(bh_data_ptr)*size);
        memset(bh_vcache,0,sizeof(bh_data_ptr)*size);
        bh_vcache_bytes  = (bh_intp*)malloc(sizeof(bh_intp)*size);
        memset(bh_vcache_bytes,0,sizeof(bh_intp)*size);
    } else {        // Disabled
        bh_vcache = NULL;
        bh_vcache_bytes = NULL;
    }

    bh_vcache_reset_counters();
}

/**
 * Deallocates vcache arrays.
 */
void bh_vcache_delete()
{
    if (0 < bh_vcache_size) {
        free(bh_vcache);
        free(bh_vcache_bytes);
    }
}

/**
 * Remove all entries from vcache and de-allocate them
 */
void bh_vcache_clear()
{
    int i;
    for (i=0; i<bh_vcache_size; i++) {
        if (bh_vcache_bytes[i] > 0) {
            bh_memory_free(bh_vcache[i], bh_vcache_bytes[i]);
        }
    }
}

/**
 * Return and remove a pointer of size 'size' from vcache.
 * NOTE: This is a helper function; it should not be used by a vector engine.
 * NOTE: This removes it from the vcache!
 *
 * @return null If none exists.
 */
bh_data_ptr bh_vcache_find(bh_intp bytes)
{
    int i;
    for (i=0; i<bh_vcache_size; i++) {
        if (bh_vcache_bytes[i] == bytes) {
            DEBUG_PRINT("Found %p in vcache.\n", bh_vcache[i]);
            bh_vcache_hits++;
            bh_vcache_bytes[i] = 0;
            bh_vcache_bytes_total -= bytes;

            return bh_vcache[i];
        }
    }
    bh_vcache_miss++;
    return NULL;
}

/**
 * Add an element to vcache.
 * NOTE: This is a helper function; it should not be used by a vector engine.
 * 
 * @param data Pointer to allocated data.
 * @param size Size in bytes of the allocated data.
 */
void bh_vcache_insert(bh_data_ptr data, bh_intp size)
{
    if (bh_vcache_bytes[bh_vcache_cur] > 0) {
		DEBUG_PRINT("Free=%p\n", bh_vcache[bh_vcache_cur]);
        bh_memory_free( bh_vcache[bh_vcache_cur], bh_vcache_bytes[bh_vcache_cur] );
    }

    bh_vcache[bh_vcache_cur] = data;
    bh_vcache_bytes[bh_vcache_cur] = size;

    bh_vcache_cur = (bh_vcache_cur+1) % bh_vcache_size;
    bh_vcache_bytes_total += size;

    bh_vcache_store++;
}

/**
 * De-allocate memory for the output operand of an instruction.
 */
bh_error bh_vcache_free(bh_instruction* inst)
{
    bh_array* base;
    bh_intp nelements, bytes;

    base = bh_base_array(inst->operand[0]);
    
    if (NULL != base->data) {
        nelements   = bh_nelements(base->ndim, base->shape);
        bytes       = nelements * bh_type_size(base->type);
        
		DEBUG_PRINT("Deallocate=%p\n", base_data);
        if (bh_vcache_size>0) {
            bh_vcache_insert(base->data, bytes);
            bh_vcache_store++;
        } else {
            bh_memory_free(base->data, bytes);
        }
		base->data = NULL;
    }
    inst->operand[0] = NULL;

    return BH_SUCCESS;
}

/**
 *  Allocate memory for the given array.
 */
bh_error bh_vcache_malloc_op(bh_array* array)
{
    bh_intp bytes;
    bh_array* base;

    if (array == NULL) {
        printf("bh_vcache_malloc(): Cannot allocate memory for a null-pointer!\n");
        return BH_ERROR;
    }

    base = bh_base_array(array);
    if (base->data != NULL) {
        printf("bh_vcache_malloc(): Already allocated!\n");
        return BH_ERROR;
    }

    bytes = bh_array_size(base);
    if (bytes <= 0) {
        printf("bh_vcache_malloc() Cannot allocate %ld bytes!\n", bytes);
        return BH_ERROR;
    }

    if (bh_vcache_size > 0) {
        base->data = bh_vcache_find(bytes);
    }
    if (base->data == NULL) {
        base->data = bh_memory_malloc(bytes);
        if (base->data == NULL) {
            int errsv = errno;
            printf("bh_data_malloc() could not allocate a data region. "
                   "Returned error code: %s.\n", strerror(errsv));
            return BH_OUT_OF_MEMORY;
        }
    }

    return BH_SUCCESS;
}

/**
 * Allocate memory for the output operand of the given instruction using vcache.
 * A previous (now unused) memory allocation will be assigned if available.
 */
bh_error bh_vcache_malloc(bh_instruction* inst)
{
    switch (inst->opcode) {                     // Allocate memory for built-in
        case BH_NONE:                           // No memory operations for these
        case BH_DISCARD:
        case BH_SYNC:
        case BH_USERFUNC:
        case BH_FREE:
            break;
        default:                                    
            return bh_vcache_malloc_op(inst->operand[0]);
            break;
    }
    return BH_SUCCESS;
}

