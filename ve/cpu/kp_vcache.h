#ifndef __KP_VCACHE_H
#define __KP_VCACHE_H 1

#include <stdbool.h>
#include "kp.h"

#ifdef __cplusplus
extern "C" {
#endif

// NOTE: kp_vcache is forward declared in kp.h

/* Allocate an alligned contigous block of memory,
 * does not apply any initialization
 *
 * @size  The size of the allocated block
 * @return A pointer to data, and NULL on error
 */
void* kp_host_malloc(size_t nbytes);

/* Frees a previously allocated data block
 *
 * @data  The pointer returned from a call to kp_host_malloc
 * @size  The size of the allocated block
 * @return Error code from munmap.
 */
int kp_host_free(void *data, size_t size);

//typedef struct kp_vcache kp_vcache;

/**
 *  Initiate vcache to a fixed capacity.
 *
 *  Cache will hold a maximum of 'capacity' entries.
 *
 *  Expiration / cache invalidation is based on round-robin;
 *  Whenever an element is added to vcache the round-robin
 *  counter is incremented.
 */
kp_vcache* kp_vcache_create(size_t capacity);

/**
 * Destroy a victim cache.
 */
void kp_vcache_destroy(kp_vcache *vcache);

/**
 *  Allocate a buffer, possibly re-using a previously de-allocated buffer.
 */
bool kp_vcache_alloc(kp_vcache *vcache, kp_buffer *buffer);

/**
 *  De-allocate provided buffer.
 */
bool kp_vcache_free(kp_vcache *vcache, kp_buffer *buffer);

size_t kp_vcache_capacity(kp_vcache *vcache);

void kp_vcache_pprint(kp_vcache* vcache);

#ifdef __cplusplus
}
#endif

#endif /* __KP_VCACHE_H */
