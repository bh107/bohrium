/*
    kp_set - A unordered container that stores unique elements.

    Provides the functionality:

    insert(value) - O(N)
    erase(value) - O(N)
    is_member(value) - O(N)

    Where N is the capacity of the set.
    O(N) is actually worst-case when the set gets fragmented,
    due to erased elements.
    Best case would be O(M) where M is the number of elements
    currently stored in the set.

    Clearly this implementation has terrible time-complexity.
    However, it is very simple and uses a single contiguous 
    memory allocation and is intended for small values of N (50-1000).

    When the default (kp_set_default_capacity) is exceeded,
    the set will realloc with (kp_set_expand_capacity) number
    of elements.

    For its intended purpose it might just be as efficient
    as a complex implementation using buckets/hashing/trees
    due to large constant factors.
*/
#ifndef __KP_SET_H
#define __KP_SET_H 1

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

static const size_t kp_set_default_capacity = 200;
static const size_t kp_set_expand_capacity = 200;

typedef struct kp_set kp_set;

kp_set* kp_set_create(void);

bool kp_set_insert(kp_set* set, void* val);

bool kp_set_erase(kp_set* set, void* val);

bool kp_set_ismember(kp_set* set, void* val);

void kp_set_destroy(kp_set* set);

size_t kp_set_size(kp_set* set);

size_t kp_set_capacity(kp_set* set);

#ifdef __cplusplus
}
#endif

#endif /* __KP_SET_H */
