#ifndef __KP_VCACHE_H
#define __KP_VCACHE_H 1

#include <stdbool.h>
#include "kp.h"

#ifdef __cplusplus
extern "C" {
#endif

void* kp_host_malloc(int64_t nbytes);

int64_t kp_host_free(void* data, int64_t size);

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
void kp_vcache_init(int64_t size);

/**
 * Remove all entries from vcache and de-allocate them
 */
void kp_vcache_clear();

/**
 * Deallocates vcache arrays.
 */
void kp_vcache_delete();

/**
 *  Allocate a buffer, possibly re-using a previously de-allocated buffer.
 */
bool kp_vcache_malloc(kp_buffer *buffer);

/**
 *  De-allocate provided buffer.
 */
bool kp_vcache_free(kp_buffer *buffer);

#ifdef __cplusplus
}
#endif

#endif /* __KP_VCACHE_H */
