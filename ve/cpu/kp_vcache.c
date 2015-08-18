#include <malloc.h>
#include <string.h>
#include <sys/mman.h>
#include "kp_utils.h"
#include "kp_vcache.h"

//
// C-Friendly version of the vcache.
//
static void** kp_vcache;                // Vcache entries
static int64_t* kp_vcache_bytes;        // Stores number of bytes in each vcache entry

static int64_t kp_vcache_bytes_total;   // Total number of bytes in vcache entries
static int64_t kp_vcache_size;          // Number of enstries
static int64_t kp_vcache_cur;           // Currently used entry

static int64_t kp_vcache_hits = 0;      // Maintain #hits from kp_vcache_malloc
static int64_t kp_vcache_miss = 0;      // Maintain #misses from kp_vcache_malloc
static int64_t kp_vcache_store = 0;     // Maintain #inserts into vcache

/**
 *  Reset vcache counters
 */
static void kp_vcache_reset_counters() {
    kp_vcache_hits = 0;
    kp_vcache_miss = 0;
    kp_vcache_store = 0;
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
void kp_vcache_init(int64_t size)
{
    kp_vcache_size = size;
    kp_vcache_bytes_total = 0;
    kp_vcache_cur = 0;

    if (0 < size) { // Enabled
        kp_vcache = malloc(sizeof(void*)*size);
        memset(kp_vcache, 0, sizeof(void*)*size);
        kp_vcache_bytes = (int64_t*)malloc(sizeof(int64_t)*size);
        memset(kp_vcache_bytes, 0, sizeof(int64_t)*size);
    } else {        // Disabled
        kp_vcache = NULL;
        kp_vcache_bytes = NULL;
    }

    kp_vcache_reset_counters();
}

/**
 * Deallocates vcache arrays.
 */
void kp_vcache_delete()
{
    if (0 < kp_vcache_size) {
        free(kp_vcache);
        free(kp_vcache_bytes);
    }
}

/**
 * Remove all entries from vcache and de-allocate them
 */
void kp_vcache_clear()
{
    for (int64_t i=0; i< kp_vcache_size; i++) {
        if (kp_vcache_bytes[i] > 0) {
            kp_host_free(kp_vcache[i], kp_vcache_bytes[i]);
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
static void* kp_vcache_find(int64_t bytes)
{
    for (int64_t i=0; i< kp_vcache_size; i++) {
        if (kp_vcache_bytes[i] == bytes) {
            kp_vcache_hits++;
            kp_vcache_bytes[i] = 0;
            kp_vcache_bytes_total -= bytes;

            return kp_vcache[i];
        }
    }
    kp_vcache_miss++;
    return NULL;
}

/**
 * Add an element to vcache.
 * NOTE: This is a helper function; it should not be used by a vector engine.
 *
 * @param data Pointer to allocated data.
 * @param size Size in bytes of the allocated data.
 */
static void kp_vcache_insert(void* data, int64_t size)
{
    if (kp_vcache_bytes[kp_vcache_cur] > 0) {
        kp_host_free(kp_vcache[kp_vcache_cur], kp_vcache_bytes[kp_vcache_cur]);
    }

    kp_vcache[kp_vcache_cur] = data;
    kp_vcache_bytes[kp_vcache_cur] = size;

    kp_vcache_cur = (kp_vcache_cur +1) % kp_vcache_size;
    kp_vcache_bytes_total += size;

    kp_vcache_store++;
}

bool kp_vcache_free(kp_buffer *buffer)
{
    int64_t nelements, bytes;

    if (NULL != buffer->data) {
        nelements   = buffer->nelem;
        bytes       = nelements * kp_etype_nbytes((KP_ETYPE)buffer->type);

        if (kp_vcache_size >0) {
            kp_vcache_insert(buffer->data, bytes);
            kp_vcache_store++;
        } else {
            kp_host_free(buffer->data, bytes);
        }
		buffer->data = NULL;
    }

    return true;
}

bool kp_vcache_malloc(kp_buffer *buffer)
{
    if (buffer->data != NULL) {   // For convenience "true" is returned
        return true;            // when data is already allocated.
    }

    int64_t bytes = kp_buffer_nbytes(buffer);
    if (bytes <= 0) {
        fprintf(stderr, "kp_vcache_malloc() Cannot allocate %lld bytes!\n", (long long)bytes);
        return false;
    }

    if (kp_vcache_size > 0) {
        buffer->data = kp_vcache_find(bytes);
    }
    if (buffer->data == NULL) {
        buffer->data = kp_host_malloc(bytes);
        if (buffer->data == NULL) {
            printf("kp_host_malloc() could not allocate a data region.");
            return false;
        }
    }

    return true;
}

/* Allocate an alligned contigous block of memory,
 * does not apply any initialization
 *
 * @size  The size of the allocated block
 * @return A pointer to data, and NULL on error
 */
void* kp_host_malloc(int64_t size)
{
    //Allocate page-size aligned memory.
    //The MAP_PRIVATE and MAP_ANONYMOUS flags is not 100% portable. See:
    //<http://stackoverflow.com/questions/4779188/how-to-use-mmap-to-allocate-a-memory-in-heap>

    void* data = mmap(0, size, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    if (data == MAP_FAILED) {
        return NULL;
    } else {
        return data;
    }
}

/* Frees a previously allocated data block
 *
 * @data  The pointer returned from a call to kp_host_malloc
 * @size  The size of the allocated block
 * @return Error code from munmap.
 */
int64_t kp_host_free(void* data, int64_t size)
{
    return munmap(data, size);
}
