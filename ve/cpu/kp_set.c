#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "assert.h"
#include "kp_set.h"

kp_set* kp_set_create(void)
{
    kp_set* set = (kp_set*)malloc(sizeof(kp_set));
    if (set) {
        set->entries = malloc(sizeof(void*)*kp_set_default_capacity);

        if (!set->entries) {
            free(set);
            set = NULL;
        } else {
            memset(set->entries, 0, sizeof(void*)*kp_set_default_capacity);
            set->nentries = 0;
            set->capacity = kp_set_default_capacity;
        }
    }
    return set;
}

bool kp_set_insert(kp_set* set, void* val)
{
    if (NULL==val) {
        fprintf(stderr, "kp_set_insert: val == NULL.\n");
        return false;
    }

    size_t free_spot = set->capacity;           // Assume that we "push_back"
    for (size_t i=0; i<set->capacity; ++i) {
        if (set->entries[i] == val) {           // Already in the set
            return true;
        }
        if ((set->entries[i] == NULL) && 
            (free_spot == set->capacity)) {     // Reuse previously erased slot
            free_spot = i;
        }
    }

    // This should occur when we could not reuse, because then
    // free_spot == set->nentries == set->capacity since all entries are occupied.
    // So we need to increase capacity.
    // free_spot is then the first element in the expanded space.
    if (free_spot == set->capacity) {
        void** expansion = realloc(set->entries,
                                   sizeof(void*)*set->capacity + 
                                   sizeof(void*)*kp_set_expand_capacity);
        if (!expansion) {
            return false;
        }
        memset(expansion+set->capacity, 0, sizeof(void*)*kp_set_expand_capacity);
        set->entries = expansion;
        set->capacity += kp_set_expand_capacity;
    }

    set->entries[free_spot] = val;
    set->nentries++;
    return true;
}

bool kp_set_erase(kp_set* set, void* val)
{
    if (!val) {
        fprintf(stderr, "kp_set_erase: val == NULL.\n");
        return false;
    }

    for (size_t i=0; i<set->capacity; ++i) {
        if (set->entries[i] == val) {
            set->entries[i] = NULL;
            set->nentries--;
            break;
        }
    }
    return true;
}

bool kp_set_ismember(kp_set* set, void* val)
{
    if (!val) {
        fprintf(stderr, "kp_set_insert: val == NULL.\n");
        return false;
    }

    for (size_t i=0; i<set->capacity; ++i) {
        if (set->entries[i] == val) {
            return true;
        }
    }
    return false;
}

size_t kp_set_capacity(kp_set* set)
{
    return set->capacity;
}

size_t kp_set_size(kp_set* set)
{
    return set->nentries;
}

void kp_set_destroy(kp_set* set)
{
    if (set) {
        if (set->entries) {
            free(set->entries);
        }
        free(set);
    }
}

