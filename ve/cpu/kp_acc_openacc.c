#include <openacc.h>

int kp_acc_init(kp_acc* acc) {
    // TODO: write an actual routine
    acc_init(acc_device_nvidia);
    return 1;
}

void kp_acc_alloc(kp_acc* acc, kp_buffer* buffer)
{
    if (kp_acc_allocated(acc, buffer)) {    // Buffer is already allocated on device
        return;
    }
    if (NULL == buffer->data) {             // Buffer is not allocated on host
        fprintf(stderr, "kp_acc_alloc: Buffer is not allocated on host.\n");
        return;
    }

    // TODO: Call OpenACC lib
    
    acc->nbytes_allocated += buffer->nelem * kp_etype_nbytes(buffer->type); // House-keeping
    kp_set_insert(acc->buffers_allocated, buffer);   
}

void kp_acc_free(kp_acc* acc, kp_buffer* buffer)
{
    if (!kp_acc_allocated(acc, buffer)) {       // Not allocated on device
        return;
    }
    if (NULL == buffer->data) {
        fprintf(stderr, "kp_acc_free: Buffer is not allocated on host.\n");
        return;
    }

    // TODO: Call OpenACC lib

    acc->nbytes_allocated -= buffer->nelem*kp_etype_nbytes(buffer->type); // Housekeeping
    kp_set_erase(acc->buffers_allocated, buffer);
    kp_set_erase(acc->buffers_pushed, buffer);
}

void kp_acc_push(kp_acc* acc, kp_buffer* buffer)
{
    if (kp_acc_pushed(acc, buffer)) {           // Already pushed to device
        return;
    }
    if (!kp_acc_allocated(acc, buffer)) {       // Not allocated on device
        fprintf(stderr, "kp_acc_push: Buffer is not allocated on device.\n");
        return;
    }

    // TODO: Call OpenACC lib

    kp_set_insert(acc->buffers_pushed, buffer); // House-keeping
}

void kp_acc_pull(kp_acc* acc, kp_buffer* buffer)
{
    if (!kp_acc_allocated(acc, buffer)) {   // Not allocated on device
        fprintf(stderr, "kp_acc_pull: Buffer is not allocated on device.\n");
        return;
    }
    if (NULL==buffer->data) {
        fprintf(stderr, "kp_acc_pull: Buffer is not allocated on host.\n");
    }

    // TODO: Call OpenACC lib
}

