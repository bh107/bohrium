#include <kp_utils.h>

size_t kp_etype_nbytes(KP_ETYPE etype)
{
    switch(etype) {
        case KP_BOOL: return 1;
        case KP_INT8: return 1;
        case KP_INT16: return 2;
        case KP_INT32: return 4;
        case KP_INT64: return 8;
        case KP_UINT8: return 1;
        case KP_UINT16: return 2;
        case KP_UINT32: return 4;
        case KP_UINT64: return 8;
        case KP_FLOAT32: return 4;
        case KP_FLOAT64: return 8;
        case KP_COMPLEX64: return 8;
        case KP_COMPLEX128: return 16;
        case KP_PAIRLL: return 16;
    }
    return 0;
}

size_t kp_buffer_nbytes(const kp_buffer* buffer)
{
    return buffer->nelem * kp_etype_nbytes(buffer->type);
}

