#ifndef MAP_ANONYMOUS
#define MAP_ANONYMOUS MAP_ANON
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include "kp_utils.h"
#include "kp_vcache.h"

//
// C-Friendly version of the vcache.
//
struct kp_vcache {
    size_t nbytes_total;   // Total number of bytes in vcache entries
    size_t capacity;       // Number of enstries
    size_t cur;            // Currently used entry

    void** entries;        // Vcache entries
    size_t* nbytes;        // Stores number of bytes in each vcache entry
};

void* kp_host_malloc(size_t nbytes)
{
    //Allocate 'nbytes' of page-aligned memory.
    //The MAP_PRIVATE and MAP_ANONYMOUS flags is not 100% portable. See:
    //<http://stackoverflow.com/questions/4779188/how-to-use-mmap-to-allocate-a-memory-in-heap>

    void* data = mmap(0, nbytes, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    if (data == MAP_FAILED) {
        return NULL;
    } else {
        return data;
    }
}

int kp_host_free(void *data, size_t size)
{
    return munmap(data, size);
}

/**
 * Return and **remove** a pointer of 'nbytes' from vcache.
 *
 * @return null If none exists.
 */
static void* kp_vcache_find(kp_vcache* vcache, size_t nbytes)
{
    if (vcache) {
        for (size_t i=0; i<vcache->capacity; i++) {
            if (vcache->nbytes[i] == nbytes) {
                void* entry = vcache->entries[i];   // Grab it
                vcache->entries[i] = NULL;          // Remove it
                vcache->nbytes[i] = 0;
                vcache->nbytes_total -= nbytes;     // Housekeeping

                return entry;                       // Exit: Return matching entry
            }
        }
    }
    return NULL;                            // Exit: Did not find an entry
}

kp_vcache* kp_vcache_create(size_t capacity)
{
    kp_vcache* vcache = malloc(sizeof(*vcache));
    if (vcache) {
        vcache->capacity = capacity;
        vcache->nbytes_total = 0;
        vcache->cur = 0;
        vcache->entries = NULL;
        vcache->nbytes = NULL;

        if (capacity) {
            vcache->entries = malloc(sizeof(void *) * capacity);
            if (vcache->entries) {
                memset(vcache->entries, 0, sizeof(void *) * capacity);

                vcache->nbytes = malloc(sizeof(*(vcache->nbytes)) * capacity);
                if (vcache->nbytes) {
                    memset(vcache->nbytes, 0, sizeof(*(vcache->nbytes)) * capacity);
                } else {
                    fprintf(stdout,
                            "kp_vcache_create: Failed allocating nbytes. Leaking...\n");
                }
            } else {
                fprintf(stdout,
                        "kp_vcache_create: Failed allocating entries. Leaking...\n");
            }
        }
    }
    return vcache;
}

void kp_vcache_destroy(kp_vcache *vcache)
{
    if (vcache) {
        if (vcache->entries) {
            for (size_t i = 0; i < vcache->capacity; i++) {    // Deallocate the entries themselves
                if (vcache->entries[i]) {
                    kp_host_free(vcache->entries[i], vcache->nbytes[i]);
                    vcache->nbytes_total -= vcache->nbytes[i];
                }
            }
            free(vcache->entries);                              // Deallocate placeholders
            free(vcache->nbytes);
        }
        free(vcache);                                           // Deallocate the victim cache itself
    }
}

/**
 * Add an entry of to vcache.
 *
 * @param data Pointer to allocated data.
 * @param size Size in bytes of the allocated data.
 */
static void kp_vcache_insert(kp_vcache* vcache, void *data, size_t nbytes)
{
    const size_t cur = vcache->cur;

    if (vcache->entries[cur]) {                 // Remove current entry
        vcache->nbytes_total -= vcache->nbytes[cur];  // Housekeeping
        kp_host_free(vcache->entries[cur], vcache->nbytes[cur]);
    }

    vcache->entries[cur] = data;                // Add it
    vcache->nbytes[cur] = nbytes;

    vcache->cur = (cur+1) % vcache->capacity;  // Housekeeping
    vcache->nbytes_total += nbytes;
}

bool kp_vcache_free(kp_vcache* vcache, kp_buffer *buffer)
{
    if (buffer->data) {
        size_t nbytes = buffer->nelem * kp_etype_nbytes((KP_ETYPE)buffer->type);
        
        if (vcache->entries) {
            kp_vcache_insert(vcache, buffer->data, nbytes);
        } else {
            kp_host_free(buffer->data, nbytes);
        }
		buffer->data = NULL;
    }
    return true;
}

bool kp_vcache_alloc(kp_vcache *vcache, kp_buffer *buffer)
{
    if (buffer->data) {         // For convenience "true" is returned
        return true;            // when data is already allocated.
    }

    const size_t nbytes = kp_buffer_nbytes(buffer);

    if (vcache->entries) {     // Find a matching entry
        buffer->data = kp_vcache_find(vcache, nbytes);
    }

    if (!buffer->data) { // Allocate anew when no entry is found in cache or when capacity == 0
        buffer->data = kp_host_malloc(nbytes);
        if (!buffer->data) {
            printf("kp_vcache_alloc(...): Failed allocating a data region using kp_host_malloc(...).");
            return false;
        }
    }

    return true;
}

size_t kp_vcache_capacity(kp_vcache *vcache)
{
    return vcache->capacity;
}

void kp_vcache_pprint(kp_vcache* vcache)
{
    if (vcache) {
        fprintf(stdout,
                "vcache(%p)[capacity(%ld), entries(%p), nbytes(%p), nbytes_total(%ld)] {\n",
                (void*)vcache,
                vcache->capacity,
                (void*)vcache->entries,
                (void*)vcache->nbytes,
                vcache->nbytes_total);
        for(size_t i=0; i<vcache->capacity; ++i) {
            fprintf(stdout,
                    " [%ld] entry(%p), nbytes(%ld)\n",
                    i,
                    vcache->entries[i],
                    vcache->nbytes[i]
            );
        }
        fprintf(stdout, "}\n");
    } else {
        fprintf(stdout, "vcache(%p) {}\n", (void*)vcache);
    }
}
