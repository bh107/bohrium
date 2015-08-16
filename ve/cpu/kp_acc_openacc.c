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

