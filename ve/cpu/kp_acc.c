#include <stdlib.h>
#include <stdio.h>
#include "kp_set.h"
#include "kp_acc.h"

struct kp_acc {
    size_t id;
    char* text_id;
    size_t bytes_allocated;
    kp_set* buffers_allocated;
    kp_set* buffers_pushed;
};

kp_acc* kp_acc_create(size_t id)
{
    kp_acc* acc = (kp_acc*)malloc(sizeof(kp_acc));
    if (acc) {
        acc->id = id;
        acc->text_id = NULL;
        acc->bytes_allocated = 0;
        acc->buffers_allocated = kp_set_create();
        acc->buffers_pushed = kp_set_create();
    }
    return acc;
}

void kp_acc_destroy(kp_acc* acc)
{
    if (acc) {
        kp_set_destroy(acc->buffers_allocated);
        kp_set_destroy(acc->buffers_pushed);
        free(acc);
    }
}

size_t kp_acc_id(kp_acc* acc)
{
    return acc->id;
}

const char* kp_acc_text(kp_acc* acc)
{
    return acc->text_id;
}

size_t kp_acc_bytes_allocated(kp_acc* acc)
{
    return acc->bytes_allocated;    
}

bool kp_acc_allocated(kp_acc* acc, kp_buffer* buffer)
{
    return kp_set_ismember(acc->buffers_allocated, buffer);
}

bool kp_acc_pushed(kp_acc* acc, kp_buffer* buffer)
{
    return kp_set_ismember(acc->buffers_pushed, buffer);
}

#if defined(CAPE_WITH_INTEL_LEO)
int kp_acc_init(kp_acc* acc)
{
    const size_t id = acc->id;
    int mthreads;
    #pragma offload target(mic:id) out(mthreads)
    {                                    
        mthreads = omp_get_max_threads();
    }                                    
    return mthreads;                     
}
#include "kp_acc_leo_autogen.c"
#else
int kp_acc_init(kp_acc* acc) { return 0; }
void kp_acc_alloc(kp_acc* acc, kp_buffer* buffer) {}
void kp_acc_free(kp_acc* acc, kp_buffer* buffer) {}
void kp_acc_push(kp_acc* acc, kp_buffer* buffer) {}
void kp_acc_pull(kp_acc* acc, kp_buffer* buffer) {}
#endif

