#include <stdio.h>
#include "kp_utils.h"

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

size_t kp_tac_noperands(const kp_tac* tac)
{
    const KP_OPERATION op = tac->op;
    const KP_OPERATOR oper = tac->oper;

    switch(op) {
        case KP_MAP:
            return 2;
        case KP_ZIP:
            return 3;
        case KP_SCAN:
            return 3;
        case KP_REDUCE_COMPLETE:
            return 3;
        case KP_REDUCE_PARTIAL:
            return 3;

        case KP_GENERATE:
            switch(oper) {
                case KP_FLOOD:
                    return 2;
                case KP_RANDOM:
                    return 3;
                case KP_RANGE:
                    return 1;
                default:
                    fprintf(stderr, "kp_tac_noperands: Unknown #operands of KP_GENEREATE, assuming 1;\n");
                    return 1;
            }
        case KP_INDEX:
            return 3;
        case KP_SYSTEM:
            switch(oper) {
                case KP_DISCARD:
                case KP_FREE:
                case KP_SYNC:
                    return 1;
                case KP_NONE:
                case KP_TALLY:
                case KP_REPEAT:
                    return 0;
                default:
                    fprintf(stderr, "kp_tac_noperands: Unknown #operands of KP_SYSTEM, assuming 0;\n");
                    return 0;
            }
        case KP_EXTENSION:
            return 3;
        case KP_NOOP:
            return 0;
    }
    return 0;
}
