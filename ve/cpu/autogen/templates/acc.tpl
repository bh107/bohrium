#compiler-settings
directiveStartToken= %
#end compiler-settings
%slurp

//
//  Allocation routines
//

%for $CTYPE, $ETYPE in $ETYPES
static void acc_alloc_${ETYPE}(kp_acc* acc, kp_buffer* buffer)
{
    const int id = kp_acc_id(acc); // Grab the accelerator device id
    ${CTYPE}* data = (${CTYPE}*)(buffer->data); // Grab the buffer and cast it
    const int nelem = buffer->nelem; // Grab # elements

    #pragma offload_transfer                                    \
            target(mic:id)                                      \
            nocopy(data:length(nelem) alloc_if(1) free_if(0))
}
%end for


//
// Free routines
//

%for $CTYPE, $ETYPE in $ETYPES
static void acc_free_${ETYPE}(kp_acc* acc, kp_buffer* buffer)
{
    const int id = kp_acc_id(acc); // Grab the accelerator device id
    ${CTYPE}* data = (${CTYPE}*)(buffer->data); // Grab the buffer and cast it
    const int nelem = buffer->nelem; // Grab # elements

    #pragma offload_transfer                                    \
            target(mic:id)                                      \
            nocopy(data:length(nelem) alloc_if(0) free_if(1))
}
%end for


//
// Push routines
//
%for $CTYPE, $ETYPE in $ETYPES
static void acc_push_${ETYPE}(kp_acc* acc, kp_buffer* buffer)
{
    const int id = kp_acc_id(acc); // Grab the accelerator device id
    ${CTYPE}* data = (${CTYPE}*)(buffer->data); // Grab the buffer and cast it
    const int nelem = buffer->nelem; // Grab # elements

    #pragma offload_transfer                                    \
            target(mic:id)                                      \
            in(data:length(nelem) alloc_if(0) free_if(0))
}
%end for

//
// Pull routines
//
%for $CTYPE, $ETYPE in $ETYPES
static void acc_pull_${ETYPE}(kp_acc* acc, kp_buffer* buffer)
{
    const int id = kp_acc_id(acc); // Grab the accelerator device id
    ${CTYPE}* data = (${CTYPE}*)(buffer->data); // Grab the buffer and cast it
    const int nelem = buffer->nelem; // Grab # elements

    #pragma offload_transfer                                    \
            target(mic:id)                                      \
            out(data:length(nelem) alloc_if(0) free_if(0))
}
%end for

void kp_acc_alloc(kp_acc* acc, kp_buffer* buffer)
{
    if (kp_acc_allocated(acc, buffer)) {    // Already allocated on device
        return;
    }
    if (NULL == buffer->data) {             // Not allocated on host!
        fprintf(stderr, "kp_acc_alloc: Buffer is not allocated on host.\n");
        return;
    }

    size_t nbytes = buffer->nelem * kp_etype_nbytes(buffer->type);
    switch(buffer->type) {                  // LEO - allocate on device
        %for $CTYPE, $ETYPE in $ETYPES
        case $ETYPE:
            acc_alloc_${ETYPE}(acc, buffer);
            break;
        %end for

        case KP_COMPLEX64:
        case KP_COMPLEX128:
        case KP_PAIRLL:
        default:
            fprintf(stderr, "kp_acc_alloc: Unsupported datatype (complex or pair).");
            break;
    }

    acc->nbytes_allocated += nbytes;        // House-keeping
    kp_set_insert(acc->buffers_allocated, buffer);   
}

void kp_acc_free(kp_acc* acc, kp_buffer* buffer)
{
    if (!kp_acc_allocated(acc, buffer)) {   // Not allocated on device!
        return;
    }
    if (NULL == buffer->data) {             // Not allocated on host!
        fprintf(stderr, "kp_acc_free: Buffer is not allocated on host.\n");
        return;
    }

    size_t nbytes = buffer->nelem * kp_etype_nbytes(buffer->type);
    switch(buffer->type) {                  // LEO - deallocate on device
        %for $CTYPE, $ETYPE in $ETYPES
        case $ETYPE:
            acc_free_${ETYPE}(acc, buffer);
            break;
        %end for

        case KP_COMPLEX64:
        case KP_COMPLEX128:
        case KP_PAIRLL:
        default:
            fprintf(stderr, "kp_acc_free: Unsupported datatype (complex or pair).");
            break;
    }

    acc->nbytes_allocated -= nbytes;        // Housekeeping
    kp_set_erase(acc->buffers_allocated, buffer);
    kp_set_erase(acc->buffers_pushed, buffer);
}

void kp_acc_push(kp_acc* acc, kp_buffer* buffer)
{
    if (kp_acc_pushed(acc, buffer)) {           // Already pushed to device
        return;
    }
    if (!kp_acc_allocated(acc, buffer)) {       // Not allocated on device!
        fprintf(stderr, "kp_acc_push: Buffer is not allocated on device.\n");
        return;
    }

    switch(buffer->type) {                      // LEO - Host ==> Device
        %for $CTYPE, $ETYPE in $ETYPES
        case $ETYPE:
            acc_push_${ETYPE}(acc, buffer);
            break;
        %end for

        case KP_COMPLEX64:
        case KP_COMPLEX128:
        case KP_PAIRLL:
        default:
            fprintf(stderr, "kp_acc_push: Unsupported datatype (complex or pair).");
            break;
    }

    kp_set_insert(acc->buffers_pushed, buffer); // House-keeping
}

void kp_acc_pull(kp_acc* acc, kp_buffer* buffer)
{
    if (!kp_acc_allocated(acc, buffer)) {       // Not allocated on device!
        fprintf(stderr, "kp_acc_pull: Buffer is not allocated on device.\n");
        return;
    }
    if (NULL==buffer->data) {                   // Not allocated on host!
        fprintf(stderr, "kp_acc_pull: Buffer is not allocated on host.\n");
    }

    switch(buffer->type) {                      // LEO - Device ==> Host
        %for $CTYPE, $ETYPE in $ETYPES
        case $ETYPE: acc_pull_${ETYPE}(acc, buffer); break;
        %end for

        case KP_COMPLEX64:
        case KP_COMPLEX128:
        case KP_PAIRLL:
        default:
            fprintf(stderr, "kp_acc_pull: Unsupported datatype (complex or pair).");
            break;
    }
}

