
#include <cphvb.h>
#include "dispatch.h"
#include "traverse.hpp"
#include "functors.hpp"

inline cphvb_error dispatch( cphvb_instruction *instr ) {

    cphvb_error res = CPHVB_SUCCESS;

    switch(instr->opcode) {

        case CPHVB_NONE:        // Nothing to do since we only use main memory.
        case CPHVB_DISCARD:
        case CPHVB_RELEASE:
        case CPHVB_SYNC:
            break;

        default:                // Element-wise functions + Memory Functions

            const long int poly = instr->opcode*100 + instr->operand[0]->type;

            switch(poly) {

                                
                case CPHVB_ADD*100+CPHVB_BOOL:
                    traverse_3<cphvb_bool, add_functor<cphvb_bool,cphvb_bool,cphvb_bool> >( instr );
                    break;
                case CPHVB_ADD*100+CPHVB_INT8:
                    traverse_3<cphvb_int8, add_functor<cphvb_int8,cphvb_int8,cphvb_int8> >( instr );
                    break;
                case CPHVB_ADD*100+CPHVB_INT16:
                    traverse_3<cphvb_int16, add_functor<cphvb_int16,cphvb_int16,cphvb_int16> >( instr );
                    break;
                case CPHVB_ADD*100+CPHVB_INT32:
                    traverse_3<cphvb_int32, add_functor<cphvb_int32,cphvb_int32,cphvb_int32> >( instr );
                    break;
                case CPHVB_ADD*100+CPHVB_INT64:
                    traverse_3<cphvb_int64, add_functor<cphvb_int64,cphvb_int64,cphvb_int64> >( instr );
                    break;
                case CPHVB_ADD*100+CPHVB_UINT8:
                    traverse_3<cphvb_uint8, add_functor<cphvb_uint8,cphvb_uint8,cphvb_uint8> >( instr );
                    break;
                case CPHVB_ADD*100+CPHVB_UINT16:
                    traverse_3<cphvb_uint16, add_functor<cphvb_uint16,cphvb_uint16,cphvb_uint16> >( instr );
                    break;
                case CPHVB_ADD*100+CPHVB_UINT32:
                    traverse_3<cphvb_uint32, add_functor<cphvb_uint32,cphvb_uint32,cphvb_uint32> >( instr );
                    break;
                case CPHVB_ADD*100+CPHVB_UINT64:
                    traverse_3<cphvb_uint64, add_functor<cphvb_uint64,cphvb_uint64,cphvb_uint64> >( instr );
                    break;
                case CPHVB_ADD*100+CPHVB_FLOAT32:
                    traverse_3<cphvb_float32, add_functor<cphvb_float32,cphvb_float32,cphvb_float32> >( instr );
                    break;
                case CPHVB_ADD*100+CPHVB_FLOAT64:
                    traverse_3<cphvb_float64, add_functor<cphvb_float64,cphvb_float64,cphvb_float64> >( instr );
                    break;
                case CPHVB_SUBTRACT*100+CPHVB_BOOL:
                    traverse_3<cphvb_bool, subtract_functor<cphvb_bool,cphvb_bool,cphvb_bool> >( instr );
                    break;
                case CPHVB_SUBTRACT*100+CPHVB_INT8:
                    traverse_3<cphvb_int8, subtract_functor<cphvb_int8,cphvb_int8,cphvb_int8> >( instr );
                    break;
                case CPHVB_SUBTRACT*100+CPHVB_INT16:
                    traverse_3<cphvb_int16, subtract_functor<cphvb_int16,cphvb_int16,cphvb_int16> >( instr );
                    break;
                case CPHVB_SUBTRACT*100+CPHVB_INT32:
                    traverse_3<cphvb_int32, subtract_functor<cphvb_int32,cphvb_int32,cphvb_int32> >( instr );
                    break;
                case CPHVB_SUBTRACT*100+CPHVB_INT64:
                    traverse_3<cphvb_int64, subtract_functor<cphvb_int64,cphvb_int64,cphvb_int64> >( instr );
                    break;
                case CPHVB_SUBTRACT*100+CPHVB_UINT8:
                    traverse_3<cphvb_uint8, subtract_functor<cphvb_uint8,cphvb_uint8,cphvb_uint8> >( instr );
                    break;
                case CPHVB_SUBTRACT*100+CPHVB_UINT16:
                    traverse_3<cphvb_uint16, subtract_functor<cphvb_uint16,cphvb_uint16,cphvb_uint16> >( instr );
                    break;
                case CPHVB_SUBTRACT*100+CPHVB_UINT32:
                    traverse_3<cphvb_uint32, subtract_functor<cphvb_uint32,cphvb_uint32,cphvb_uint32> >( instr );
                    break;
                case CPHVB_SUBTRACT*100+CPHVB_UINT64:
                    traverse_3<cphvb_uint64, subtract_functor<cphvb_uint64,cphvb_uint64,cphvb_uint64> >( instr );
                    break;
                case CPHVB_SUBTRACT*100+CPHVB_FLOAT32:
                    traverse_3<cphvb_float32, subtract_functor<cphvb_float32,cphvb_float32,cphvb_float32> >( instr );
                    break;
                case CPHVB_SUBTRACT*100+CPHVB_FLOAT64:
                    traverse_3<cphvb_float64, subtract_functor<cphvb_float64,cphvb_float64,cphvb_float64> >( instr );
                    break;
                case CPHVB_MULTIPLY*100+CPHVB_BOOL:
                    traverse_3<cphvb_bool, multiply_functor<cphvb_bool,cphvb_bool,cphvb_bool> >( instr );
                    break;
                case CPHVB_MULTIPLY*100+CPHVB_INT8:
                    traverse_3<cphvb_int8, multiply_functor<cphvb_int8,cphvb_int8,cphvb_int8> >( instr );
                    break;
                case CPHVB_MULTIPLY*100+CPHVB_INT16:
                    traverse_3<cphvb_int16, multiply_functor<cphvb_int16,cphvb_int16,cphvb_int16> >( instr );
                    break;
                case CPHVB_MULTIPLY*100+CPHVB_INT32:
                    traverse_3<cphvb_int32, multiply_functor<cphvb_int32,cphvb_int32,cphvb_int32> >( instr );
                    break;
                case CPHVB_MULTIPLY*100+CPHVB_INT64:
                    traverse_3<cphvb_int64, multiply_functor<cphvb_int64,cphvb_int64,cphvb_int64> >( instr );
                    break;
                case CPHVB_MULTIPLY*100+CPHVB_UINT8:
                    traverse_3<cphvb_uint8, multiply_functor<cphvb_uint8,cphvb_uint8,cphvb_uint8> >( instr );
                    break;
                case CPHVB_MULTIPLY*100+CPHVB_UINT16:
                    traverse_3<cphvb_uint16, multiply_functor<cphvb_uint16,cphvb_uint16,cphvb_uint16> >( instr );
                    break;
                case CPHVB_MULTIPLY*100+CPHVB_UINT32:
                    traverse_3<cphvb_uint32, multiply_functor<cphvb_uint32,cphvb_uint32,cphvb_uint32> >( instr );
                    break;
                case CPHVB_MULTIPLY*100+CPHVB_UINT64:
                    traverse_3<cphvb_uint64, multiply_functor<cphvb_uint64,cphvb_uint64,cphvb_uint64> >( instr );
                    break;
                case CPHVB_MULTIPLY*100+CPHVB_FLOAT32:
                    traverse_3<cphvb_float32, multiply_functor<cphvb_float32,cphvb_float32,cphvb_float32> >( instr );
                    break;
                case CPHVB_MULTIPLY*100+CPHVB_FLOAT64:
                    traverse_3<cphvb_float64, multiply_functor<cphvb_float64,cphvb_float64,cphvb_float64> >( instr );
                    break;
                case CPHVB_DIVIDE*100+CPHVB_BOOL:
                    traverse_3<cphvb_bool, divide_functor<cphvb_bool,cphvb_bool,cphvb_bool> >( instr );
                    break;
                case CPHVB_DIVIDE*100+CPHVB_INT8:
                    traverse_3<cphvb_int8, divide_functor<cphvb_int8,cphvb_int8,cphvb_int8> >( instr );
                    break;
                case CPHVB_DIVIDE*100+CPHVB_INT16:
                    traverse_3<cphvb_int16, divide_functor<cphvb_int16,cphvb_int16,cphvb_int16> >( instr );
                    break;
                case CPHVB_DIVIDE*100+CPHVB_INT32:
                    traverse_3<cphvb_int32, divide_functor<cphvb_int32,cphvb_int32,cphvb_int32> >( instr );
                    break;
                case CPHVB_DIVIDE*100+CPHVB_INT64:
                    traverse_3<cphvb_int64, divide_functor<cphvb_int64,cphvb_int64,cphvb_int64> >( instr );
                    break;
                case CPHVB_DIVIDE*100+CPHVB_UINT8:
                    traverse_3<cphvb_uint8, divide_functor<cphvb_uint8,cphvb_uint8,cphvb_uint8> >( instr );
                    break;
                case CPHVB_DIVIDE*100+CPHVB_UINT16:
                    traverse_3<cphvb_uint16, divide_functor<cphvb_uint16,cphvb_uint16,cphvb_uint16> >( instr );
                    break;
                case CPHVB_DIVIDE*100+CPHVB_UINT32:
                    traverse_3<cphvb_uint32, divide_functor<cphvb_uint32,cphvb_uint32,cphvb_uint32> >( instr );
                    break;
                case CPHVB_DIVIDE*100+CPHVB_UINT64:
                    traverse_3<cphvb_uint64, divide_functor<cphvb_uint64,cphvb_uint64,cphvb_uint64> >( instr );
                    break;
                case CPHVB_DIVIDE*100+CPHVB_FLOAT32:
                    traverse_3<cphvb_float32, divide_functor<cphvb_float32,cphvb_float32,cphvb_float32> >( instr );
                    break;
                case CPHVB_DIVIDE*100+CPHVB_FLOAT64:
                    traverse_3<cphvb_float64, divide_functor<cphvb_float64,cphvb_float64,cphvb_float64> >( instr );
                    break;
                case CPHVB_LOGADDEXP*100+CPHVB_BOOL:
                    traverse_3<cphvb_bool, logaddexp_functor<cphvb_bool,cphvb_bool,cphvb_bool> >( instr );
                    break;
                case CPHVB_LOGADDEXP*100+CPHVB_INT8:
                    traverse_3<cphvb_int8, logaddexp_functor<cphvb_int8,cphvb_int8,cphvb_int8> >( instr );
                    break;
                case CPHVB_LOGADDEXP*100+CPHVB_INT16:
                    traverse_3<cphvb_int16, logaddexp_functor<cphvb_int16,cphvb_int16,cphvb_int16> >( instr );
                    break;
                case CPHVB_LOGADDEXP*100+CPHVB_INT32:
                    traverse_3<cphvb_int32, logaddexp_functor<cphvb_int32,cphvb_int32,cphvb_int32> >( instr );
                    break;
                case CPHVB_LOGADDEXP*100+CPHVB_INT64:
                    traverse_3<cphvb_int64, logaddexp_functor<cphvb_int64,cphvb_int64,cphvb_int64> >( instr );
                    break;
                case CPHVB_LOGADDEXP*100+CPHVB_UINT8:
                    traverse_3<cphvb_uint8, logaddexp_functor<cphvb_uint8,cphvb_uint8,cphvb_uint8> >( instr );
                    break;
                case CPHVB_LOGADDEXP*100+CPHVB_UINT16:
                    traverse_3<cphvb_uint16, logaddexp_functor<cphvb_uint16,cphvb_uint16,cphvb_uint16> >( instr );
                    break;
                case CPHVB_LOGADDEXP*100+CPHVB_UINT32:
                    traverse_3<cphvb_uint32, logaddexp_functor<cphvb_uint32,cphvb_uint32,cphvb_uint32> >( instr );
                    break;
                case CPHVB_LOGADDEXP*100+CPHVB_UINT64:
                    traverse_3<cphvb_uint64, logaddexp_functor<cphvb_uint64,cphvb_uint64,cphvb_uint64> >( instr );
                    break;
                case CPHVB_LOGADDEXP*100+CPHVB_FLOAT32:
                    traverse_3<cphvb_float32, logaddexp_functor<cphvb_float32,cphvb_float32,cphvb_float32> >( instr );
                    break;
                case CPHVB_LOGADDEXP*100+CPHVB_FLOAT64:
                    traverse_3<cphvb_float64, logaddexp_functor<cphvb_float64,cphvb_float64,cphvb_float64> >( instr );
                    break;
                case CPHVB_LOGADDEXP2*100+CPHVB_BOOL:
                    traverse_3<cphvb_bool, logaddexp2_functor<cphvb_bool,cphvb_bool,cphvb_bool> >( instr );
                    break;
                case CPHVB_LOGADDEXP2*100+CPHVB_INT8:
                    traverse_3<cphvb_int8, logaddexp2_functor<cphvb_int8,cphvb_int8,cphvb_int8> >( instr );
                    break;
                case CPHVB_LOGADDEXP2*100+CPHVB_INT16:
                    traverse_3<cphvb_int16, logaddexp2_functor<cphvb_int16,cphvb_int16,cphvb_int16> >( instr );
                    break;
                case CPHVB_LOGADDEXP2*100+CPHVB_INT32:
                    traverse_3<cphvb_int32, logaddexp2_functor<cphvb_int32,cphvb_int32,cphvb_int32> >( instr );
                    break;
                case CPHVB_LOGADDEXP2*100+CPHVB_INT64:
                    traverse_3<cphvb_int64, logaddexp2_functor<cphvb_int64,cphvb_int64,cphvb_int64> >( instr );
                    break;
                case CPHVB_LOGADDEXP2*100+CPHVB_UINT8:
                    traverse_3<cphvb_uint8, logaddexp2_functor<cphvb_uint8,cphvb_uint8,cphvb_uint8> >( instr );
                    break;
                case CPHVB_LOGADDEXP2*100+CPHVB_UINT16:
                    traverse_3<cphvb_uint16, logaddexp2_functor<cphvb_uint16,cphvb_uint16,cphvb_uint16> >( instr );
                    break;
                case CPHVB_LOGADDEXP2*100+CPHVB_UINT32:
                    traverse_3<cphvb_uint32, logaddexp2_functor<cphvb_uint32,cphvb_uint32,cphvb_uint32> >( instr );
                    break;
                case CPHVB_LOGADDEXP2*100+CPHVB_UINT64:
                    traverse_3<cphvb_uint64, logaddexp2_functor<cphvb_uint64,cphvb_uint64,cphvb_uint64> >( instr );
                    break;
                case CPHVB_LOGADDEXP2*100+CPHVB_FLOAT32:
                    traverse_3<cphvb_float32, logaddexp2_functor<cphvb_float32,cphvb_float32,cphvb_float32> >( instr );
                    break;
                case CPHVB_LOGADDEXP2*100+CPHVB_FLOAT64:
                    traverse_3<cphvb_float64, logaddexp2_functor<cphvb_float64,cphvb_float64,cphvb_float64> >( instr );
                    break;
                case CPHVB_TRUE_DIVIDE*100+CPHVB_BOOL:
                    traverse_3<cphvb_bool, true_divide_functor<cphvb_bool,cphvb_bool,cphvb_bool> >( instr );
                    break;
                case CPHVB_TRUE_DIVIDE*100+CPHVB_INT8:
                    traverse_3<cphvb_int8, true_divide_functor<cphvb_int8,cphvb_int8,cphvb_int8> >( instr );
                    break;
                case CPHVB_TRUE_DIVIDE*100+CPHVB_INT16:
                    traverse_3<cphvb_int16, true_divide_functor<cphvb_int16,cphvb_int16,cphvb_int16> >( instr );
                    break;
                case CPHVB_TRUE_DIVIDE*100+CPHVB_INT32:
                    traverse_3<cphvb_int32, true_divide_functor<cphvb_int32,cphvb_int32,cphvb_int32> >( instr );
                    break;
                case CPHVB_TRUE_DIVIDE*100+CPHVB_INT64:
                    traverse_3<cphvb_int64, true_divide_functor<cphvb_int64,cphvb_int64,cphvb_int64> >( instr );
                    break;
                case CPHVB_TRUE_DIVIDE*100+CPHVB_UINT8:
                    traverse_3<cphvb_uint8, true_divide_functor<cphvb_uint8,cphvb_uint8,cphvb_uint8> >( instr );
                    break;
                case CPHVB_TRUE_DIVIDE*100+CPHVB_UINT16:
                    traverse_3<cphvb_uint16, true_divide_functor<cphvb_uint16,cphvb_uint16,cphvb_uint16> >( instr );
                    break;
                case CPHVB_TRUE_DIVIDE*100+CPHVB_UINT32:
                    traverse_3<cphvb_uint32, true_divide_functor<cphvb_uint32,cphvb_uint32,cphvb_uint32> >( instr );
                    break;
                case CPHVB_TRUE_DIVIDE*100+CPHVB_UINT64:
                    traverse_3<cphvb_uint64, true_divide_functor<cphvb_uint64,cphvb_uint64,cphvb_uint64> >( instr );
                    break;
                case CPHVB_TRUE_DIVIDE*100+CPHVB_FLOAT32:
                    traverse_3<cphvb_float32, true_divide_functor<cphvb_float32,cphvb_float32,cphvb_float32> >( instr );
                    break;
                case CPHVB_TRUE_DIVIDE*100+CPHVB_FLOAT64:
                    traverse_3<cphvb_float64, true_divide_functor<cphvb_float64,cphvb_float64,cphvb_float64> >( instr );
                    break;
                case CPHVB_FLOOR_DIVIDE*100+CPHVB_BOOL:
                    traverse_3<cphvb_bool, floor_divide_functor<cphvb_bool,cphvb_bool,cphvb_bool> >( instr );
                    break;
                case CPHVB_FLOOR_DIVIDE*100+CPHVB_INT8:
                    traverse_3<cphvb_int8, floor_divide_functor<cphvb_int8,cphvb_int8,cphvb_int8> >( instr );
                    break;
                case CPHVB_FLOOR_DIVIDE*100+CPHVB_INT16:
                    traverse_3<cphvb_int16, floor_divide_functor<cphvb_int16,cphvb_int16,cphvb_int16> >( instr );
                    break;
                case CPHVB_FLOOR_DIVIDE*100+CPHVB_INT32:
                    traverse_3<cphvb_int32, floor_divide_functor<cphvb_int32,cphvb_int32,cphvb_int32> >( instr );
                    break;
                case CPHVB_FLOOR_DIVIDE*100+CPHVB_INT64:
                    traverse_3<cphvb_int64, floor_divide_functor<cphvb_int64,cphvb_int64,cphvb_int64> >( instr );
                    break;
                case CPHVB_FLOOR_DIVIDE*100+CPHVB_UINT8:
                    traverse_3<cphvb_uint8, floor_divide_functor<cphvb_uint8,cphvb_uint8,cphvb_uint8> >( instr );
                    break;
                case CPHVB_FLOOR_DIVIDE*100+CPHVB_UINT16:
                    traverse_3<cphvb_uint16, floor_divide_functor<cphvb_uint16,cphvb_uint16,cphvb_uint16> >( instr );
                    break;
                case CPHVB_FLOOR_DIVIDE*100+CPHVB_UINT32:
                    traverse_3<cphvb_uint32, floor_divide_functor<cphvb_uint32,cphvb_uint32,cphvb_uint32> >( instr );
                    break;
                case CPHVB_FLOOR_DIVIDE*100+CPHVB_UINT64:
                    traverse_3<cphvb_uint64, floor_divide_functor<cphvb_uint64,cphvb_uint64,cphvb_uint64> >( instr );
                    break;
                case CPHVB_FLOOR_DIVIDE*100+CPHVB_FLOAT32:
                    traverse_3<cphvb_float32, floor_divide_functor<cphvb_float32,cphvb_float32,cphvb_float32> >( instr );
                    break;
                case CPHVB_FLOOR_DIVIDE*100+CPHVB_FLOAT64:
                    traverse_3<cphvb_float64, floor_divide_functor<cphvb_float64,cphvb_float64,cphvb_float64> >( instr );
                    break;
                case CPHVB_POWER*100+CPHVB_BOOL:
                    traverse_3<cphvb_bool, power_functor<cphvb_bool,cphvb_bool,cphvb_bool> >( instr );
                    break;
                case CPHVB_POWER*100+CPHVB_INT8:
                    traverse_3<cphvb_int8, power_functor<cphvb_int8,cphvb_int8,cphvb_int8> >( instr );
                    break;
                case CPHVB_POWER*100+CPHVB_INT16:
                    traverse_3<cphvb_int16, power_functor<cphvb_int16,cphvb_int16,cphvb_int16> >( instr );
                    break;
                case CPHVB_POWER*100+CPHVB_INT32:
                    traverse_3<cphvb_int32, power_functor<cphvb_int32,cphvb_int32,cphvb_int32> >( instr );
                    break;
                case CPHVB_POWER*100+CPHVB_INT64:
                    traverse_3<cphvb_int64, power_functor<cphvb_int64,cphvb_int64,cphvb_int64> >( instr );
                    break;
                case CPHVB_POWER*100+CPHVB_UINT8:
                    traverse_3<cphvb_uint8, power_functor<cphvb_uint8,cphvb_uint8,cphvb_uint8> >( instr );
                    break;
                case CPHVB_POWER*100+CPHVB_UINT16:
                    traverse_3<cphvb_uint16, power_functor<cphvb_uint16,cphvb_uint16,cphvb_uint16> >( instr );
                    break;
                case CPHVB_POWER*100+CPHVB_UINT32:
                    traverse_3<cphvb_uint32, power_functor<cphvb_uint32,cphvb_uint32,cphvb_uint32> >( instr );
                    break;
                case CPHVB_POWER*100+CPHVB_UINT64:
                    traverse_3<cphvb_uint64, power_functor<cphvb_uint64,cphvb_uint64,cphvb_uint64> >( instr );
                    break;
                case CPHVB_POWER*100+CPHVB_FLOAT32:
                    traverse_3<cphvb_float32, power_functor<cphvb_float32,cphvb_float32,cphvb_float32> >( instr );
                    break;
                case CPHVB_POWER*100+CPHVB_FLOAT64:
                    traverse_3<cphvb_float64, power_functor<cphvb_float64,cphvb_float64,cphvb_float64> >( instr );
                    break;
                case CPHVB_REMAINDER*100+CPHVB_BOOL:
                    traverse_3<cphvb_bool, remainder_functor<cphvb_bool,cphvb_bool,cphvb_bool> >( instr );
                    break;
                case CPHVB_REMAINDER*100+CPHVB_INT8:
                    traverse_3<cphvb_int8, remainder_functor<cphvb_int8,cphvb_int8,cphvb_int8> >( instr );
                    break;
                case CPHVB_REMAINDER*100+CPHVB_INT16:
                    traverse_3<cphvb_int16, remainder_functor<cphvb_int16,cphvb_int16,cphvb_int16> >( instr );
                    break;
                case CPHVB_REMAINDER*100+CPHVB_INT32:
                    traverse_3<cphvb_int32, remainder_functor<cphvb_int32,cphvb_int32,cphvb_int32> >( instr );
                    break;
                case CPHVB_REMAINDER*100+CPHVB_INT64:
                    traverse_3<cphvb_int64, remainder_functor<cphvb_int64,cphvb_int64,cphvb_int64> >( instr );
                    break;
                case CPHVB_REMAINDER*100+CPHVB_UINT8:
                    traverse_3<cphvb_uint8, remainder_functor<cphvb_uint8,cphvb_uint8,cphvb_uint8> >( instr );
                    break;
                case CPHVB_REMAINDER*100+CPHVB_UINT16:
                    traverse_3<cphvb_uint16, remainder_functor<cphvb_uint16,cphvb_uint16,cphvb_uint16> >( instr );
                    break;
                case CPHVB_REMAINDER*100+CPHVB_UINT32:
                    traverse_3<cphvb_uint32, remainder_functor<cphvb_uint32,cphvb_uint32,cphvb_uint32> >( instr );
                    break;
                case CPHVB_REMAINDER*100+CPHVB_UINT64:
                    traverse_3<cphvb_uint64, remainder_functor<cphvb_uint64,cphvb_uint64,cphvb_uint64> >( instr );
                    break;
                case CPHVB_REMAINDER*100+CPHVB_FLOAT32:
                    traverse_3<cphvb_float32, remainder_functor<cphvb_float32,cphvb_float32,cphvb_float32> >( instr );
                    break;
                case CPHVB_REMAINDER*100+CPHVB_FLOAT64:
                    traverse_3<cphvb_float64, remainder_functor<cphvb_float64,cphvb_float64,cphvb_float64> >( instr );
                    break;
                case CPHVB_MOD*100+CPHVB_BOOL:
                    traverse_3<cphvb_bool, mod_functor<cphvb_bool,cphvb_bool,cphvb_bool> >( instr );
                    break;
                case CPHVB_MOD*100+CPHVB_INT8:
                    traverse_3<cphvb_int8, mod_functor<cphvb_int8,cphvb_int8,cphvb_int8> >( instr );
                    break;
                case CPHVB_MOD*100+CPHVB_INT16:
                    traverse_3<cphvb_int16, mod_functor<cphvb_int16,cphvb_int16,cphvb_int16> >( instr );
                    break;
                case CPHVB_MOD*100+CPHVB_INT32:
                    traverse_3<cphvb_int32, mod_functor<cphvb_int32,cphvb_int32,cphvb_int32> >( instr );
                    break;
                case CPHVB_MOD*100+CPHVB_INT64:
                    traverse_3<cphvb_int64, mod_functor<cphvb_int64,cphvb_int64,cphvb_int64> >( instr );
                    break;
                case CPHVB_MOD*100+CPHVB_UINT8:
                    traverse_3<cphvb_uint8, mod_functor<cphvb_uint8,cphvb_uint8,cphvb_uint8> >( instr );
                    break;
                case CPHVB_MOD*100+CPHVB_UINT16:
                    traverse_3<cphvb_uint16, mod_functor<cphvb_uint16,cphvb_uint16,cphvb_uint16> >( instr );
                    break;
                case CPHVB_MOD*100+CPHVB_UINT32:
                    traverse_3<cphvb_uint32, mod_functor<cphvb_uint32,cphvb_uint32,cphvb_uint32> >( instr );
                    break;
                case CPHVB_MOD*100+CPHVB_UINT64:
                    traverse_3<cphvb_uint64, mod_functor<cphvb_uint64,cphvb_uint64,cphvb_uint64> >( instr );
                    break;
                case CPHVB_MOD*100+CPHVB_FLOAT32:
                    traverse_3<cphvb_float32, mod_functor<cphvb_float32,cphvb_float32,cphvb_float32> >( instr );
                    break;
                case CPHVB_MOD*100+CPHVB_FLOAT64:
                    traverse_3<cphvb_float64, mod_functor<cphvb_float64,cphvb_float64,cphvb_float64> >( instr );
                    break;
                case CPHVB_FMOD*100+CPHVB_BOOL:
                    traverse_3<cphvb_bool, fmod_functor<cphvb_bool,cphvb_bool,cphvb_bool> >( instr );
                    break;
                case CPHVB_FMOD*100+CPHVB_INT8:
                    traverse_3<cphvb_int8, fmod_functor<cphvb_int8,cphvb_int8,cphvb_int8> >( instr );
                    break;
                case CPHVB_FMOD*100+CPHVB_INT16:
                    traverse_3<cphvb_int16, fmod_functor<cphvb_int16,cphvb_int16,cphvb_int16> >( instr );
                    break;
                case CPHVB_FMOD*100+CPHVB_INT32:
                    traverse_3<cphvb_int32, fmod_functor<cphvb_int32,cphvb_int32,cphvb_int32> >( instr );
                    break;
                case CPHVB_FMOD*100+CPHVB_INT64:
                    traverse_3<cphvb_int64, fmod_functor<cphvb_int64,cphvb_int64,cphvb_int64> >( instr );
                    break;
                case CPHVB_FMOD*100+CPHVB_UINT8:
                    traverse_3<cphvb_uint8, fmod_functor<cphvb_uint8,cphvb_uint8,cphvb_uint8> >( instr );
                    break;
                case CPHVB_FMOD*100+CPHVB_UINT16:
                    traverse_3<cphvb_uint16, fmod_functor<cphvb_uint16,cphvb_uint16,cphvb_uint16> >( instr );
                    break;
                case CPHVB_FMOD*100+CPHVB_UINT32:
                    traverse_3<cphvb_uint32, fmod_functor<cphvb_uint32,cphvb_uint32,cphvb_uint32> >( instr );
                    break;
                case CPHVB_FMOD*100+CPHVB_UINT64:
                    traverse_3<cphvb_uint64, fmod_functor<cphvb_uint64,cphvb_uint64,cphvb_uint64> >( instr );
                    break;
                case CPHVB_FMOD*100+CPHVB_FLOAT32:
                    traverse_3<cphvb_float32, fmod_functor<cphvb_float32,cphvb_float32,cphvb_float32> >( instr );
                    break;
                case CPHVB_FMOD*100+CPHVB_FLOAT64:
                    traverse_3<cphvb_float64, fmod_functor<cphvb_float64,cphvb_float64,cphvb_float64> >( instr );
                    break;
                case CPHVB_BITWISE_AND*100+CPHVB_BOOL:
                    traverse_3<cphvb_bool, bitwise_and_functor<cphvb_bool,cphvb_bool,cphvb_bool> >( instr );
                    break;
                case CPHVB_BITWISE_AND*100+CPHVB_INT8:
                    traverse_3<cphvb_int8, bitwise_and_functor<cphvb_int8,cphvb_int8,cphvb_int8> >( instr );
                    break;
                case CPHVB_BITWISE_AND*100+CPHVB_INT16:
                    traverse_3<cphvb_int16, bitwise_and_functor<cphvb_int16,cphvb_int16,cphvb_int16> >( instr );
                    break;
                case CPHVB_BITWISE_AND*100+CPHVB_INT32:
                    traverse_3<cphvb_int32, bitwise_and_functor<cphvb_int32,cphvb_int32,cphvb_int32> >( instr );
                    break;
                case CPHVB_BITWISE_AND*100+CPHVB_INT64:
                    traverse_3<cphvb_int64, bitwise_and_functor<cphvb_int64,cphvb_int64,cphvb_int64> >( instr );
                    break;
                case CPHVB_BITWISE_AND*100+CPHVB_UINT8:
                    traverse_3<cphvb_uint8, bitwise_and_functor<cphvb_uint8,cphvb_uint8,cphvb_uint8> >( instr );
                    break;
                case CPHVB_BITWISE_AND*100+CPHVB_UINT16:
                    traverse_3<cphvb_uint16, bitwise_and_functor<cphvb_uint16,cphvb_uint16,cphvb_uint16> >( instr );
                    break;
                case CPHVB_BITWISE_AND*100+CPHVB_UINT32:
                    traverse_3<cphvb_uint32, bitwise_and_functor<cphvb_uint32,cphvb_uint32,cphvb_uint32> >( instr );
                    break;
                case CPHVB_BITWISE_AND*100+CPHVB_UINT64:
                    traverse_3<cphvb_uint64, bitwise_and_functor<cphvb_uint64,cphvb_uint64,cphvb_uint64> >( instr );
                    break;
                case CPHVB_BITWISE_AND*100+CPHVB_FLOAT32:
                    traverse_3<cphvb_float32, bitwise_and_functor<cphvb_float32,cphvb_float32,cphvb_float32> >( instr );
                    break;
                case CPHVB_BITWISE_AND*100+CPHVB_FLOAT64:
                    traverse_3<cphvb_float64, bitwise_and_functor<cphvb_float64,cphvb_float64,cphvb_float64> >( instr );
                    break;
                case CPHVB_BITWISE_OR*100+CPHVB_BOOL:
                    traverse_3<cphvb_bool, bitwise_or_functor<cphvb_bool,cphvb_bool,cphvb_bool> >( instr );
                    break;
                case CPHVB_BITWISE_OR*100+CPHVB_INT8:
                    traverse_3<cphvb_int8, bitwise_or_functor<cphvb_int8,cphvb_int8,cphvb_int8> >( instr );
                    break;
                case CPHVB_BITWISE_OR*100+CPHVB_INT16:
                    traverse_3<cphvb_int16, bitwise_or_functor<cphvb_int16,cphvb_int16,cphvb_int16> >( instr );
                    break;
                case CPHVB_BITWISE_OR*100+CPHVB_INT32:
                    traverse_3<cphvb_int32, bitwise_or_functor<cphvb_int32,cphvb_int32,cphvb_int32> >( instr );
                    break;
                case CPHVB_BITWISE_OR*100+CPHVB_INT64:
                    traverse_3<cphvb_int64, bitwise_or_functor<cphvb_int64,cphvb_int64,cphvb_int64> >( instr );
                    break;
                case CPHVB_BITWISE_OR*100+CPHVB_UINT8:
                    traverse_3<cphvb_uint8, bitwise_or_functor<cphvb_uint8,cphvb_uint8,cphvb_uint8> >( instr );
                    break;
                case CPHVB_BITWISE_OR*100+CPHVB_UINT16:
                    traverse_3<cphvb_uint16, bitwise_or_functor<cphvb_uint16,cphvb_uint16,cphvb_uint16> >( instr );
                    break;
                case CPHVB_BITWISE_OR*100+CPHVB_UINT32:
                    traverse_3<cphvb_uint32, bitwise_or_functor<cphvb_uint32,cphvb_uint32,cphvb_uint32> >( instr );
                    break;
                case CPHVB_BITWISE_OR*100+CPHVB_UINT64:
                    traverse_3<cphvb_uint64, bitwise_or_functor<cphvb_uint64,cphvb_uint64,cphvb_uint64> >( instr );
                    break;
                case CPHVB_BITWISE_OR*100+CPHVB_FLOAT32:
                    traverse_3<cphvb_float32, bitwise_or_functor<cphvb_float32,cphvb_float32,cphvb_float32> >( instr );
                    break;
                case CPHVB_BITWISE_OR*100+CPHVB_FLOAT64:
                    traverse_3<cphvb_float64, bitwise_or_functor<cphvb_float64,cphvb_float64,cphvb_float64> >( instr );
                    break;
                case CPHVB_BITWISE_XOR*100+CPHVB_BOOL:
                    traverse_3<cphvb_bool, bitwise_xor_functor<cphvb_bool,cphvb_bool,cphvb_bool> >( instr );
                    break;
                case CPHVB_BITWISE_XOR*100+CPHVB_INT8:
                    traverse_3<cphvb_int8, bitwise_xor_functor<cphvb_int8,cphvb_int8,cphvb_int8> >( instr );
                    break;
                case CPHVB_BITWISE_XOR*100+CPHVB_INT16:
                    traverse_3<cphvb_int16, bitwise_xor_functor<cphvb_int16,cphvb_int16,cphvb_int16> >( instr );
                    break;
                case CPHVB_BITWISE_XOR*100+CPHVB_INT32:
                    traverse_3<cphvb_int32, bitwise_xor_functor<cphvb_int32,cphvb_int32,cphvb_int32> >( instr );
                    break;
                case CPHVB_BITWISE_XOR*100+CPHVB_INT64:
                    traverse_3<cphvb_int64, bitwise_xor_functor<cphvb_int64,cphvb_int64,cphvb_int64> >( instr );
                    break;
                case CPHVB_BITWISE_XOR*100+CPHVB_UINT8:
                    traverse_3<cphvb_uint8, bitwise_xor_functor<cphvb_uint8,cphvb_uint8,cphvb_uint8> >( instr );
                    break;
                case CPHVB_BITWISE_XOR*100+CPHVB_UINT16:
                    traverse_3<cphvb_uint16, bitwise_xor_functor<cphvb_uint16,cphvb_uint16,cphvb_uint16> >( instr );
                    break;
                case CPHVB_BITWISE_XOR*100+CPHVB_UINT32:
                    traverse_3<cphvb_uint32, bitwise_xor_functor<cphvb_uint32,cphvb_uint32,cphvb_uint32> >( instr );
                    break;
                case CPHVB_BITWISE_XOR*100+CPHVB_UINT64:
                    traverse_3<cphvb_uint64, bitwise_xor_functor<cphvb_uint64,cphvb_uint64,cphvb_uint64> >( instr );
                    break;
                case CPHVB_BITWISE_XOR*100+CPHVB_FLOAT32:
                    traverse_3<cphvb_float32, bitwise_xor_functor<cphvb_float32,cphvb_float32,cphvb_float32> >( instr );
                    break;
                case CPHVB_BITWISE_XOR*100+CPHVB_FLOAT64:
                    traverse_3<cphvb_float64, bitwise_xor_functor<cphvb_float64,cphvb_float64,cphvb_float64> >( instr );
                    break;
                case CPHVB_LOGICAL_AND*100+CPHVB_BOOL:
                    traverse_3<cphvb_bool, logical_and_functor<cphvb_bool,cphvb_bool,cphvb_bool> >( instr );
                    break;
                case CPHVB_LOGICAL_AND*100+CPHVB_INT8:
                    traverse_3<cphvb_int8, logical_and_functor<cphvb_int8,cphvb_int8,cphvb_int8> >( instr );
                    break;
                case CPHVB_LOGICAL_AND*100+CPHVB_INT16:
                    traverse_3<cphvb_int16, logical_and_functor<cphvb_int16,cphvb_int16,cphvb_int16> >( instr );
                    break;
                case CPHVB_LOGICAL_AND*100+CPHVB_INT32:
                    traverse_3<cphvb_int32, logical_and_functor<cphvb_int32,cphvb_int32,cphvb_int32> >( instr );
                    break;
                case CPHVB_LOGICAL_AND*100+CPHVB_INT64:
                    traverse_3<cphvb_int64, logical_and_functor<cphvb_int64,cphvb_int64,cphvb_int64> >( instr );
                    break;
                case CPHVB_LOGICAL_AND*100+CPHVB_UINT8:
                    traverse_3<cphvb_uint8, logical_and_functor<cphvb_uint8,cphvb_uint8,cphvb_uint8> >( instr );
                    break;
                case CPHVB_LOGICAL_AND*100+CPHVB_UINT16:
                    traverse_3<cphvb_uint16, logical_and_functor<cphvb_uint16,cphvb_uint16,cphvb_uint16> >( instr );
                    break;
                case CPHVB_LOGICAL_AND*100+CPHVB_UINT32:
                    traverse_3<cphvb_uint32, logical_and_functor<cphvb_uint32,cphvb_uint32,cphvb_uint32> >( instr );
                    break;
                case CPHVB_LOGICAL_AND*100+CPHVB_UINT64:
                    traverse_3<cphvb_uint64, logical_and_functor<cphvb_uint64,cphvb_uint64,cphvb_uint64> >( instr );
                    break;
                case CPHVB_LOGICAL_AND*100+CPHVB_FLOAT32:
                    traverse_3<cphvb_float32, logical_and_functor<cphvb_float32,cphvb_float32,cphvb_float32> >( instr );
                    break;
                case CPHVB_LOGICAL_AND*100+CPHVB_FLOAT64:
                    traverse_3<cphvb_float64, logical_and_functor<cphvb_float64,cphvb_float64,cphvb_float64> >( instr );
                    break;
                case CPHVB_LOGICAL_OR*100+CPHVB_BOOL:
                    traverse_3<cphvb_bool, logical_or_functor<cphvb_bool,cphvb_bool,cphvb_bool> >( instr );
                    break;
                case CPHVB_LOGICAL_OR*100+CPHVB_INT8:
                    traverse_3<cphvb_int8, logical_or_functor<cphvb_int8,cphvb_int8,cphvb_int8> >( instr );
                    break;
                case CPHVB_LOGICAL_OR*100+CPHVB_INT16:
                    traverse_3<cphvb_int16, logical_or_functor<cphvb_int16,cphvb_int16,cphvb_int16> >( instr );
                    break;
                case CPHVB_LOGICAL_OR*100+CPHVB_INT32:
                    traverse_3<cphvb_int32, logical_or_functor<cphvb_int32,cphvb_int32,cphvb_int32> >( instr );
                    break;
                case CPHVB_LOGICAL_OR*100+CPHVB_INT64:
                    traverse_3<cphvb_int64, logical_or_functor<cphvb_int64,cphvb_int64,cphvb_int64> >( instr );
                    break;
                case CPHVB_LOGICAL_OR*100+CPHVB_UINT8:
                    traverse_3<cphvb_uint8, logical_or_functor<cphvb_uint8,cphvb_uint8,cphvb_uint8> >( instr );
                    break;
                case CPHVB_LOGICAL_OR*100+CPHVB_UINT16:
                    traverse_3<cphvb_uint16, logical_or_functor<cphvb_uint16,cphvb_uint16,cphvb_uint16> >( instr );
                    break;
                case CPHVB_LOGICAL_OR*100+CPHVB_UINT32:
                    traverse_3<cphvb_uint32, logical_or_functor<cphvb_uint32,cphvb_uint32,cphvb_uint32> >( instr );
                    break;
                case CPHVB_LOGICAL_OR*100+CPHVB_UINT64:
                    traverse_3<cphvb_uint64, logical_or_functor<cphvb_uint64,cphvb_uint64,cphvb_uint64> >( instr );
                    break;
                case CPHVB_LOGICAL_OR*100+CPHVB_FLOAT32:
                    traverse_3<cphvb_float32, logical_or_functor<cphvb_float32,cphvb_float32,cphvb_float32> >( instr );
                    break;
                case CPHVB_LOGICAL_OR*100+CPHVB_FLOAT64:
                    traverse_3<cphvb_float64, logical_or_functor<cphvb_float64,cphvb_float64,cphvb_float64> >( instr );
                    break;
                case CPHVB_LOGICAL_XOR*100+CPHVB_BOOL:
                    traverse_3<cphvb_bool, logical_xor_functor<cphvb_bool,cphvb_bool,cphvb_bool> >( instr );
                    break;
                case CPHVB_LOGICAL_XOR*100+CPHVB_INT8:
                    traverse_3<cphvb_int8, logical_xor_functor<cphvb_int8,cphvb_int8,cphvb_int8> >( instr );
                    break;
                case CPHVB_LOGICAL_XOR*100+CPHVB_INT16:
                    traverse_3<cphvb_int16, logical_xor_functor<cphvb_int16,cphvb_int16,cphvb_int16> >( instr );
                    break;
                case CPHVB_LOGICAL_XOR*100+CPHVB_INT32:
                    traverse_3<cphvb_int32, logical_xor_functor<cphvb_int32,cphvb_int32,cphvb_int32> >( instr );
                    break;
                case CPHVB_LOGICAL_XOR*100+CPHVB_INT64:
                    traverse_3<cphvb_int64, logical_xor_functor<cphvb_int64,cphvb_int64,cphvb_int64> >( instr );
                    break;
                case CPHVB_LOGICAL_XOR*100+CPHVB_UINT8:
                    traverse_3<cphvb_uint8, logical_xor_functor<cphvb_uint8,cphvb_uint8,cphvb_uint8> >( instr );
                    break;
                case CPHVB_LOGICAL_XOR*100+CPHVB_UINT16:
                    traverse_3<cphvb_uint16, logical_xor_functor<cphvb_uint16,cphvb_uint16,cphvb_uint16> >( instr );
                    break;
                case CPHVB_LOGICAL_XOR*100+CPHVB_UINT32:
                    traverse_3<cphvb_uint32, logical_xor_functor<cphvb_uint32,cphvb_uint32,cphvb_uint32> >( instr );
                    break;
                case CPHVB_LOGICAL_XOR*100+CPHVB_UINT64:
                    traverse_3<cphvb_uint64, logical_xor_functor<cphvb_uint64,cphvb_uint64,cphvb_uint64> >( instr );
                    break;
                case CPHVB_LOGICAL_XOR*100+CPHVB_FLOAT32:
                    traverse_3<cphvb_float32, logical_xor_functor<cphvb_float32,cphvb_float32,cphvb_float32> >( instr );
                    break;
                case CPHVB_LOGICAL_XOR*100+CPHVB_FLOAT64:
                    traverse_3<cphvb_float64, logical_xor_functor<cphvb_float64,cphvb_float64,cphvb_float64> >( instr );
                    break;
                case CPHVB_LEFT_SHIFT*100+CPHVB_BOOL:
                    traverse_3<cphvb_bool, left_shift_functor<cphvb_bool,cphvb_bool,cphvb_bool> >( instr );
                    break;
                case CPHVB_LEFT_SHIFT*100+CPHVB_INT8:
                    traverse_3<cphvb_int8, left_shift_functor<cphvb_int8,cphvb_int8,cphvb_int8> >( instr );
                    break;
                case CPHVB_LEFT_SHIFT*100+CPHVB_INT16:
                    traverse_3<cphvb_int16, left_shift_functor<cphvb_int16,cphvb_int16,cphvb_int16> >( instr );
                    break;
                case CPHVB_LEFT_SHIFT*100+CPHVB_INT32:
                    traverse_3<cphvb_int32, left_shift_functor<cphvb_int32,cphvb_int32,cphvb_int32> >( instr );
                    break;
                case CPHVB_LEFT_SHIFT*100+CPHVB_INT64:
                    traverse_3<cphvb_int64, left_shift_functor<cphvb_int64,cphvb_int64,cphvb_int64> >( instr );
                    break;
                case CPHVB_LEFT_SHIFT*100+CPHVB_UINT8:
                    traverse_3<cphvb_uint8, left_shift_functor<cphvb_uint8,cphvb_uint8,cphvb_uint8> >( instr );
                    break;
                case CPHVB_LEFT_SHIFT*100+CPHVB_UINT16:
                    traverse_3<cphvb_uint16, left_shift_functor<cphvb_uint16,cphvb_uint16,cphvb_uint16> >( instr );
                    break;
                case CPHVB_LEFT_SHIFT*100+CPHVB_UINT32:
                    traverse_3<cphvb_uint32, left_shift_functor<cphvb_uint32,cphvb_uint32,cphvb_uint32> >( instr );
                    break;
                case CPHVB_LEFT_SHIFT*100+CPHVB_UINT64:
                    traverse_3<cphvb_uint64, left_shift_functor<cphvb_uint64,cphvb_uint64,cphvb_uint64> >( instr );
                    break;
                case CPHVB_LEFT_SHIFT*100+CPHVB_FLOAT32:
                    traverse_3<cphvb_float32, left_shift_functor<cphvb_float32,cphvb_float32,cphvb_float32> >( instr );
                    break;
                case CPHVB_LEFT_SHIFT*100+CPHVB_FLOAT64:
                    traverse_3<cphvb_float64, left_shift_functor<cphvb_float64,cphvb_float64,cphvb_float64> >( instr );
                    break;
                case CPHVB_RIGHT_SHIFT*100+CPHVB_BOOL:
                    traverse_3<cphvb_bool, right_shift_functor<cphvb_bool,cphvb_bool,cphvb_bool> >( instr );
                    break;
                case CPHVB_RIGHT_SHIFT*100+CPHVB_INT8:
                    traverse_3<cphvb_int8, right_shift_functor<cphvb_int8,cphvb_int8,cphvb_int8> >( instr );
                    break;
                case CPHVB_RIGHT_SHIFT*100+CPHVB_INT16:
                    traverse_3<cphvb_int16, right_shift_functor<cphvb_int16,cphvb_int16,cphvb_int16> >( instr );
                    break;
                case CPHVB_RIGHT_SHIFT*100+CPHVB_INT32:
                    traverse_3<cphvb_int32, right_shift_functor<cphvb_int32,cphvb_int32,cphvb_int32> >( instr );
                    break;
                case CPHVB_RIGHT_SHIFT*100+CPHVB_INT64:
                    traverse_3<cphvb_int64, right_shift_functor<cphvb_int64,cphvb_int64,cphvb_int64> >( instr );
                    break;
                case CPHVB_RIGHT_SHIFT*100+CPHVB_UINT8:
                    traverse_3<cphvb_uint8, right_shift_functor<cphvb_uint8,cphvb_uint8,cphvb_uint8> >( instr );
                    break;
                case CPHVB_RIGHT_SHIFT*100+CPHVB_UINT16:
                    traverse_3<cphvb_uint16, right_shift_functor<cphvb_uint16,cphvb_uint16,cphvb_uint16> >( instr );
                    break;
                case CPHVB_RIGHT_SHIFT*100+CPHVB_UINT32:
                    traverse_3<cphvb_uint32, right_shift_functor<cphvb_uint32,cphvb_uint32,cphvb_uint32> >( instr );
                    break;
                case CPHVB_RIGHT_SHIFT*100+CPHVB_UINT64:
                    traverse_3<cphvb_uint64, right_shift_functor<cphvb_uint64,cphvb_uint64,cphvb_uint64> >( instr );
                    break;
                case CPHVB_RIGHT_SHIFT*100+CPHVB_FLOAT32:
                    traverse_3<cphvb_float32, right_shift_functor<cphvb_float32,cphvb_float32,cphvb_float32> >( instr );
                    break;
                case CPHVB_RIGHT_SHIFT*100+CPHVB_FLOAT64:
                    traverse_3<cphvb_float64, right_shift_functor<cphvb_float64,cphvb_float64,cphvb_float64> >( instr );
                    break;
                case CPHVB_GREATER*100+CPHVB_BOOL:
                    traverse_3<cphvb_bool, greater_functor<cphvb_bool,cphvb_bool,cphvb_bool> >( instr );
                    break;
                case CPHVB_GREATER*100+CPHVB_INT8:
                    traverse_3<cphvb_int8, greater_functor<cphvb_int8,cphvb_int8,cphvb_int8> >( instr );
                    break;
                case CPHVB_GREATER*100+CPHVB_INT16:
                    traverse_3<cphvb_int16, greater_functor<cphvb_int16,cphvb_int16,cphvb_int16> >( instr );
                    break;
                case CPHVB_GREATER*100+CPHVB_INT32:
                    traverse_3<cphvb_int32, greater_functor<cphvb_int32,cphvb_int32,cphvb_int32> >( instr );
                    break;
                case CPHVB_GREATER*100+CPHVB_INT64:
                    traverse_3<cphvb_int64, greater_functor<cphvb_int64,cphvb_int64,cphvb_int64> >( instr );
                    break;
                case CPHVB_GREATER*100+CPHVB_UINT8:
                    traverse_3<cphvb_uint8, greater_functor<cphvb_uint8,cphvb_uint8,cphvb_uint8> >( instr );
                    break;
                case CPHVB_GREATER*100+CPHVB_UINT16:
                    traverse_3<cphvb_uint16, greater_functor<cphvb_uint16,cphvb_uint16,cphvb_uint16> >( instr );
                    break;
                case CPHVB_GREATER*100+CPHVB_UINT32:
                    traverse_3<cphvb_uint32, greater_functor<cphvb_uint32,cphvb_uint32,cphvb_uint32> >( instr );
                    break;
                case CPHVB_GREATER*100+CPHVB_UINT64:
                    traverse_3<cphvb_uint64, greater_functor<cphvb_uint64,cphvb_uint64,cphvb_uint64> >( instr );
                    break;
                case CPHVB_GREATER*100+CPHVB_FLOAT32:
                    traverse_3<cphvb_float32, greater_functor<cphvb_float32,cphvb_float32,cphvb_float32> >( instr );
                    break;
                case CPHVB_GREATER*100+CPHVB_FLOAT64:
                    traverse_3<cphvb_float64, greater_functor<cphvb_float64,cphvb_float64,cphvb_float64> >( instr );
                    break;
                case CPHVB_GREATER_EQUAL*100+CPHVB_BOOL:
                    traverse_3<cphvb_bool, greater_equal_functor<cphvb_bool,cphvb_bool,cphvb_bool> >( instr );
                    break;
                case CPHVB_GREATER_EQUAL*100+CPHVB_INT8:
                    traverse_3<cphvb_int8, greater_equal_functor<cphvb_int8,cphvb_int8,cphvb_int8> >( instr );
                    break;
                case CPHVB_GREATER_EQUAL*100+CPHVB_INT16:
                    traverse_3<cphvb_int16, greater_equal_functor<cphvb_int16,cphvb_int16,cphvb_int16> >( instr );
                    break;
                case CPHVB_GREATER_EQUAL*100+CPHVB_INT32:
                    traverse_3<cphvb_int32, greater_equal_functor<cphvb_int32,cphvb_int32,cphvb_int32> >( instr );
                    break;
                case CPHVB_GREATER_EQUAL*100+CPHVB_INT64:
                    traverse_3<cphvb_int64, greater_equal_functor<cphvb_int64,cphvb_int64,cphvb_int64> >( instr );
                    break;
                case CPHVB_GREATER_EQUAL*100+CPHVB_UINT8:
                    traverse_3<cphvb_uint8, greater_equal_functor<cphvb_uint8,cphvb_uint8,cphvb_uint8> >( instr );
                    break;
                case CPHVB_GREATER_EQUAL*100+CPHVB_UINT16:
                    traverse_3<cphvb_uint16, greater_equal_functor<cphvb_uint16,cphvb_uint16,cphvb_uint16> >( instr );
                    break;
                case CPHVB_GREATER_EQUAL*100+CPHVB_UINT32:
                    traverse_3<cphvb_uint32, greater_equal_functor<cphvb_uint32,cphvb_uint32,cphvb_uint32> >( instr );
                    break;
                case CPHVB_GREATER_EQUAL*100+CPHVB_UINT64:
                    traverse_3<cphvb_uint64, greater_equal_functor<cphvb_uint64,cphvb_uint64,cphvb_uint64> >( instr );
                    break;
                case CPHVB_GREATER_EQUAL*100+CPHVB_FLOAT32:
                    traverse_3<cphvb_float32, greater_equal_functor<cphvb_float32,cphvb_float32,cphvb_float32> >( instr );
                    break;
                case CPHVB_GREATER_EQUAL*100+CPHVB_FLOAT64:
                    traverse_3<cphvb_float64, greater_equal_functor<cphvb_float64,cphvb_float64,cphvb_float64> >( instr );
                    break;
                case CPHVB_LESS*100+CPHVB_BOOL:
                    traverse_3<cphvb_bool, less_functor<cphvb_bool,cphvb_bool,cphvb_bool> >( instr );
                    break;
                case CPHVB_LESS*100+CPHVB_INT8:
                    traverse_3<cphvb_int8, less_functor<cphvb_int8,cphvb_int8,cphvb_int8> >( instr );
                    break;
                case CPHVB_LESS*100+CPHVB_INT16:
                    traverse_3<cphvb_int16, less_functor<cphvb_int16,cphvb_int16,cphvb_int16> >( instr );
                    break;
                case CPHVB_LESS*100+CPHVB_INT32:
                    traverse_3<cphvb_int32, less_functor<cphvb_int32,cphvb_int32,cphvb_int32> >( instr );
                    break;
                case CPHVB_LESS*100+CPHVB_INT64:
                    traverse_3<cphvb_int64, less_functor<cphvb_int64,cphvb_int64,cphvb_int64> >( instr );
                    break;
                case CPHVB_LESS*100+CPHVB_UINT8:
                    traverse_3<cphvb_uint8, less_functor<cphvb_uint8,cphvb_uint8,cphvb_uint8> >( instr );
                    break;
                case CPHVB_LESS*100+CPHVB_UINT16:
                    traverse_3<cphvb_uint16, less_functor<cphvb_uint16,cphvb_uint16,cphvb_uint16> >( instr );
                    break;
                case CPHVB_LESS*100+CPHVB_UINT32:
                    traverse_3<cphvb_uint32, less_functor<cphvb_uint32,cphvb_uint32,cphvb_uint32> >( instr );
                    break;
                case CPHVB_LESS*100+CPHVB_UINT64:
                    traverse_3<cphvb_uint64, less_functor<cphvb_uint64,cphvb_uint64,cphvb_uint64> >( instr );
                    break;
                case CPHVB_LESS*100+CPHVB_FLOAT32:
                    traverse_3<cphvb_float32, less_functor<cphvb_float32,cphvb_float32,cphvb_float32> >( instr );
                    break;
                case CPHVB_LESS*100+CPHVB_FLOAT64:
                    traverse_3<cphvb_float64, less_functor<cphvb_float64,cphvb_float64,cphvb_float64> >( instr );
                    break;
                case CPHVB_LESS_EQUAL*100+CPHVB_BOOL:
                    traverse_3<cphvb_bool, less_equal_functor<cphvb_bool,cphvb_bool,cphvb_bool> >( instr );
                    break;
                case CPHVB_LESS_EQUAL*100+CPHVB_INT8:
                    traverse_3<cphvb_int8, less_equal_functor<cphvb_int8,cphvb_int8,cphvb_int8> >( instr );
                    break;
                case CPHVB_LESS_EQUAL*100+CPHVB_INT16:
                    traverse_3<cphvb_int16, less_equal_functor<cphvb_int16,cphvb_int16,cphvb_int16> >( instr );
                    break;
                case CPHVB_LESS_EQUAL*100+CPHVB_INT32:
                    traverse_3<cphvb_int32, less_equal_functor<cphvb_int32,cphvb_int32,cphvb_int32> >( instr );
                    break;
                case CPHVB_LESS_EQUAL*100+CPHVB_INT64:
                    traverse_3<cphvb_int64, less_equal_functor<cphvb_int64,cphvb_int64,cphvb_int64> >( instr );
                    break;
                case CPHVB_LESS_EQUAL*100+CPHVB_UINT8:
                    traverse_3<cphvb_uint8, less_equal_functor<cphvb_uint8,cphvb_uint8,cphvb_uint8> >( instr );
                    break;
                case CPHVB_LESS_EQUAL*100+CPHVB_UINT16:
                    traverse_3<cphvb_uint16, less_equal_functor<cphvb_uint16,cphvb_uint16,cphvb_uint16> >( instr );
                    break;
                case CPHVB_LESS_EQUAL*100+CPHVB_UINT32:
                    traverse_3<cphvb_uint32, less_equal_functor<cphvb_uint32,cphvb_uint32,cphvb_uint32> >( instr );
                    break;
                case CPHVB_LESS_EQUAL*100+CPHVB_UINT64:
                    traverse_3<cphvb_uint64, less_equal_functor<cphvb_uint64,cphvb_uint64,cphvb_uint64> >( instr );
                    break;
                case CPHVB_LESS_EQUAL*100+CPHVB_FLOAT32:
                    traverse_3<cphvb_float32, less_equal_functor<cphvb_float32,cphvb_float32,cphvb_float32> >( instr );
                    break;
                case CPHVB_LESS_EQUAL*100+CPHVB_FLOAT64:
                    traverse_3<cphvb_float64, less_equal_functor<cphvb_float64,cphvb_float64,cphvb_float64> >( instr );
                    break;
                case CPHVB_NOT_EQUAL*100+CPHVB_BOOL:
                    traverse_3<cphvb_bool, not_equal_functor<cphvb_bool,cphvb_bool,cphvb_bool> >( instr );
                    break;
                case CPHVB_NOT_EQUAL*100+CPHVB_INT8:
                    traverse_3<cphvb_int8, not_equal_functor<cphvb_int8,cphvb_int8,cphvb_int8> >( instr );
                    break;
                case CPHVB_NOT_EQUAL*100+CPHVB_INT16:
                    traverse_3<cphvb_int16, not_equal_functor<cphvb_int16,cphvb_int16,cphvb_int16> >( instr );
                    break;
                case CPHVB_NOT_EQUAL*100+CPHVB_INT32:
                    traverse_3<cphvb_int32, not_equal_functor<cphvb_int32,cphvb_int32,cphvb_int32> >( instr );
                    break;
                case CPHVB_NOT_EQUAL*100+CPHVB_INT64:
                    traverse_3<cphvb_int64, not_equal_functor<cphvb_int64,cphvb_int64,cphvb_int64> >( instr );
                    break;
                case CPHVB_NOT_EQUAL*100+CPHVB_UINT8:
                    traverse_3<cphvb_uint8, not_equal_functor<cphvb_uint8,cphvb_uint8,cphvb_uint8> >( instr );
                    break;
                case CPHVB_NOT_EQUAL*100+CPHVB_UINT16:
                    traverse_3<cphvb_uint16, not_equal_functor<cphvb_uint16,cphvb_uint16,cphvb_uint16> >( instr );
                    break;
                case CPHVB_NOT_EQUAL*100+CPHVB_UINT32:
                    traverse_3<cphvb_uint32, not_equal_functor<cphvb_uint32,cphvb_uint32,cphvb_uint32> >( instr );
                    break;
                case CPHVB_NOT_EQUAL*100+CPHVB_UINT64:
                    traverse_3<cphvb_uint64, not_equal_functor<cphvb_uint64,cphvb_uint64,cphvb_uint64> >( instr );
                    break;
                case CPHVB_NOT_EQUAL*100+CPHVB_FLOAT32:
                    traverse_3<cphvb_float32, not_equal_functor<cphvb_float32,cphvb_float32,cphvb_float32> >( instr );
                    break;
                case CPHVB_NOT_EQUAL*100+CPHVB_FLOAT64:
                    traverse_3<cphvb_float64, not_equal_functor<cphvb_float64,cphvb_float64,cphvb_float64> >( instr );
                    break;
                case CPHVB_EQUAL*100+CPHVB_BOOL:
                    traverse_3<cphvb_bool, equal_functor<cphvb_bool,cphvb_bool,cphvb_bool> >( instr );
                    break;
                case CPHVB_EQUAL*100+CPHVB_INT8:
                    traverse_3<cphvb_int8, equal_functor<cphvb_int8,cphvb_int8,cphvb_int8> >( instr );
                    break;
                case CPHVB_EQUAL*100+CPHVB_INT16:
                    traverse_3<cphvb_int16, equal_functor<cphvb_int16,cphvb_int16,cphvb_int16> >( instr );
                    break;
                case CPHVB_EQUAL*100+CPHVB_INT32:
                    traverse_3<cphvb_int32, equal_functor<cphvb_int32,cphvb_int32,cphvb_int32> >( instr );
                    break;
                case CPHVB_EQUAL*100+CPHVB_INT64:
                    traverse_3<cphvb_int64, equal_functor<cphvb_int64,cphvb_int64,cphvb_int64> >( instr );
                    break;
                case CPHVB_EQUAL*100+CPHVB_UINT8:
                    traverse_3<cphvb_uint8, equal_functor<cphvb_uint8,cphvb_uint8,cphvb_uint8> >( instr );
                    break;
                case CPHVB_EQUAL*100+CPHVB_UINT16:
                    traverse_3<cphvb_uint16, equal_functor<cphvb_uint16,cphvb_uint16,cphvb_uint16> >( instr );
                    break;
                case CPHVB_EQUAL*100+CPHVB_UINT32:
                    traverse_3<cphvb_uint32, equal_functor<cphvb_uint32,cphvb_uint32,cphvb_uint32> >( instr );
                    break;
                case CPHVB_EQUAL*100+CPHVB_UINT64:
                    traverse_3<cphvb_uint64, equal_functor<cphvb_uint64,cphvb_uint64,cphvb_uint64> >( instr );
                    break;
                case CPHVB_EQUAL*100+CPHVB_FLOAT32:
                    traverse_3<cphvb_float32, equal_functor<cphvb_float32,cphvb_float32,cphvb_float32> >( instr );
                    break;
                case CPHVB_EQUAL*100+CPHVB_FLOAT64:
                    traverse_3<cphvb_float64, equal_functor<cphvb_float64,cphvb_float64,cphvb_float64> >( instr );
                    break;
                case CPHVB_MAXIMUM*100+CPHVB_BOOL:
                    traverse_3<cphvb_bool, maximum_functor<cphvb_bool,cphvb_bool,cphvb_bool> >( instr );
                    break;
                case CPHVB_MAXIMUM*100+CPHVB_INT8:
                    traverse_3<cphvb_int8, maximum_functor<cphvb_int8,cphvb_int8,cphvb_int8> >( instr );
                    break;
                case CPHVB_MAXIMUM*100+CPHVB_INT16:
                    traverse_3<cphvb_int16, maximum_functor<cphvb_int16,cphvb_int16,cphvb_int16> >( instr );
                    break;
                case CPHVB_MAXIMUM*100+CPHVB_INT32:
                    traverse_3<cphvb_int32, maximum_functor<cphvb_int32,cphvb_int32,cphvb_int32> >( instr );
                    break;
                case CPHVB_MAXIMUM*100+CPHVB_INT64:
                    traverse_3<cphvb_int64, maximum_functor<cphvb_int64,cphvb_int64,cphvb_int64> >( instr );
                    break;
                case CPHVB_MAXIMUM*100+CPHVB_UINT8:
                    traverse_3<cphvb_uint8, maximum_functor<cphvb_uint8,cphvb_uint8,cphvb_uint8> >( instr );
                    break;
                case CPHVB_MAXIMUM*100+CPHVB_UINT16:
                    traverse_3<cphvb_uint16, maximum_functor<cphvb_uint16,cphvb_uint16,cphvb_uint16> >( instr );
                    break;
                case CPHVB_MAXIMUM*100+CPHVB_UINT32:
                    traverse_3<cphvb_uint32, maximum_functor<cphvb_uint32,cphvb_uint32,cphvb_uint32> >( instr );
                    break;
                case CPHVB_MAXIMUM*100+CPHVB_UINT64:
                    traverse_3<cphvb_uint64, maximum_functor<cphvb_uint64,cphvb_uint64,cphvb_uint64> >( instr );
                    break;
                case CPHVB_MAXIMUM*100+CPHVB_FLOAT32:
                    traverse_3<cphvb_float32, maximum_functor<cphvb_float32,cphvb_float32,cphvb_float32> >( instr );
                    break;
                case CPHVB_MAXIMUM*100+CPHVB_FLOAT64:
                    traverse_3<cphvb_float64, maximum_functor<cphvb_float64,cphvb_float64,cphvb_float64> >( instr );
                    break;
                case CPHVB_MINIMUM*100+CPHVB_BOOL:
                    traverse_3<cphvb_bool, minimum_functor<cphvb_bool,cphvb_bool,cphvb_bool> >( instr );
                    break;
                case CPHVB_MINIMUM*100+CPHVB_INT8:
                    traverse_3<cphvb_int8, minimum_functor<cphvb_int8,cphvb_int8,cphvb_int8> >( instr );
                    break;
                case CPHVB_MINIMUM*100+CPHVB_INT16:
                    traverse_3<cphvb_int16, minimum_functor<cphvb_int16,cphvb_int16,cphvb_int16> >( instr );
                    break;
                case CPHVB_MINIMUM*100+CPHVB_INT32:
                    traverse_3<cphvb_int32, minimum_functor<cphvb_int32,cphvb_int32,cphvb_int32> >( instr );
                    break;
                case CPHVB_MINIMUM*100+CPHVB_INT64:
                    traverse_3<cphvb_int64, minimum_functor<cphvb_int64,cphvb_int64,cphvb_int64> >( instr );
                    break;
                case CPHVB_MINIMUM*100+CPHVB_UINT8:
                    traverse_3<cphvb_uint8, minimum_functor<cphvb_uint8,cphvb_uint8,cphvb_uint8> >( instr );
                    break;
                case CPHVB_MINIMUM*100+CPHVB_UINT16:
                    traverse_3<cphvb_uint16, minimum_functor<cphvb_uint16,cphvb_uint16,cphvb_uint16> >( instr );
                    break;
                case CPHVB_MINIMUM*100+CPHVB_UINT32:
                    traverse_3<cphvb_uint32, minimum_functor<cphvb_uint32,cphvb_uint32,cphvb_uint32> >( instr );
                    break;
                case CPHVB_MINIMUM*100+CPHVB_UINT64:
                    traverse_3<cphvb_uint64, minimum_functor<cphvb_uint64,cphvb_uint64,cphvb_uint64> >( instr );
                    break;
                case CPHVB_MINIMUM*100+CPHVB_FLOAT32:
                    traverse_3<cphvb_float32, minimum_functor<cphvb_float32,cphvb_float32,cphvb_float32> >( instr );
                    break;
                case CPHVB_MINIMUM*100+CPHVB_FLOAT64:
                    traverse_3<cphvb_float64, minimum_functor<cphvb_float64,cphvb_float64,cphvb_float64> >( instr );
                    break;
                case CPHVB_LDEXP*100+CPHVB_BOOL:
                    traverse_3<cphvb_bool, ldexp_functor<cphvb_bool,cphvb_bool,cphvb_bool> >( instr );
                    break;
                case CPHVB_LDEXP*100+CPHVB_INT8:
                    traverse_3<cphvb_int8, ldexp_functor<cphvb_int8,cphvb_int8,cphvb_int8> >( instr );
                    break;
                case CPHVB_LDEXP*100+CPHVB_INT16:
                    traverse_3<cphvb_int16, ldexp_functor<cphvb_int16,cphvb_int16,cphvb_int16> >( instr );
                    break;
                case CPHVB_LDEXP*100+CPHVB_INT32:
                    traverse_3<cphvb_int32, ldexp_functor<cphvb_int32,cphvb_int32,cphvb_int32> >( instr );
                    break;
                case CPHVB_LDEXP*100+CPHVB_INT64:
                    traverse_3<cphvb_int64, ldexp_functor<cphvb_int64,cphvb_int64,cphvb_int64> >( instr );
                    break;
                case CPHVB_LDEXP*100+CPHVB_UINT8:
                    traverse_3<cphvb_uint8, ldexp_functor<cphvb_uint8,cphvb_uint8,cphvb_uint8> >( instr );
                    break;
                case CPHVB_LDEXP*100+CPHVB_UINT16:
                    traverse_3<cphvb_uint16, ldexp_functor<cphvb_uint16,cphvb_uint16,cphvb_uint16> >( instr );
                    break;
                case CPHVB_LDEXP*100+CPHVB_UINT32:
                    traverse_3<cphvb_uint32, ldexp_functor<cphvb_uint32,cphvb_uint32,cphvb_uint32> >( instr );
                    break;
                case CPHVB_LDEXP*100+CPHVB_UINT64:
                    traverse_3<cphvb_uint64, ldexp_functor<cphvb_uint64,cphvb_uint64,cphvb_uint64> >( instr );
                    break;
                case CPHVB_LDEXP*100+CPHVB_FLOAT32:
                    traverse_3<cphvb_float32, ldexp_functor<cphvb_float32,cphvb_float32,cphvb_float32> >( instr );
                    break;
                case CPHVB_LDEXP*100+CPHVB_FLOAT64:
                    traverse_3<cphvb_float64, ldexp_functor<cphvb_float64,cphvb_float64,cphvb_float64> >( instr );
                    break;
                case CPHVB_NEGATIVE*100+CPHVB_BOOL:
                    traverse_2<cphvb_bool, negative_functor<cphvb_bool,cphvb_bool> >( instr );
                    break;
                case CPHVB_NEGATIVE*100+CPHVB_INT8:
                    traverse_2<cphvb_int8, negative_functor<cphvb_int8,cphvb_int8> >( instr );
                    break;
                case CPHVB_NEGATIVE*100+CPHVB_INT16:
                    traverse_2<cphvb_int16, negative_functor<cphvb_int16,cphvb_int16> >( instr );
                    break;
                case CPHVB_NEGATIVE*100+CPHVB_INT32:
                    traverse_2<cphvb_int32, negative_functor<cphvb_int32,cphvb_int32> >( instr );
                    break;
                case CPHVB_NEGATIVE*100+CPHVB_INT64:
                    traverse_2<cphvb_int64, negative_functor<cphvb_int64,cphvb_int64> >( instr );
                    break;
                case CPHVB_NEGATIVE*100+CPHVB_UINT8:
                    traverse_2<cphvb_uint8, negative_functor<cphvb_uint8,cphvb_uint8> >( instr );
                    break;
                case CPHVB_NEGATIVE*100+CPHVB_UINT16:
                    traverse_2<cphvb_uint16, negative_functor<cphvb_uint16,cphvb_uint16> >( instr );
                    break;
                case CPHVB_NEGATIVE*100+CPHVB_UINT32:
                    traverse_2<cphvb_uint32, negative_functor<cphvb_uint32,cphvb_uint32> >( instr );
                    break;
                case CPHVB_NEGATIVE*100+CPHVB_UINT64:
                    traverse_2<cphvb_uint64, negative_functor<cphvb_uint64,cphvb_uint64> >( instr );
                    break;
                case CPHVB_NEGATIVE*100+CPHVB_FLOAT32:
                    traverse_2<cphvb_float32, negative_functor<cphvb_float32,cphvb_float32> >( instr );
                    break;
                case CPHVB_NEGATIVE*100+CPHVB_FLOAT64:
                    traverse_2<cphvb_float64, negative_functor<cphvb_float64,cphvb_float64> >( instr );
                    break;
                case CPHVB_ABSOLUTE*100+CPHVB_BOOL:
                    traverse_2<cphvb_bool, absolute_functor<cphvb_bool,cphvb_bool> >( instr );
                    break;
                case CPHVB_ABSOLUTE*100+CPHVB_INT8:
                    traverse_2<cphvb_int8, absolute_functor<cphvb_int8,cphvb_int8> >( instr );
                    break;
                case CPHVB_ABSOLUTE*100+CPHVB_INT16:
                    traverse_2<cphvb_int16, absolute_functor<cphvb_int16,cphvb_int16> >( instr );
                    break;
                case CPHVB_ABSOLUTE*100+CPHVB_INT32:
                    traverse_2<cphvb_int32, absolute_functor<cphvb_int32,cphvb_int32> >( instr );
                    break;
                case CPHVB_ABSOLUTE*100+CPHVB_INT64:
                    traverse_2<cphvb_int64, absolute_functor<cphvb_int64,cphvb_int64> >( instr );
                    break;
                case CPHVB_ABSOLUTE*100+CPHVB_UINT8:
                    traverse_2<cphvb_uint8, absolute_functor<cphvb_uint8,cphvb_uint8> >( instr );
                    break;
                case CPHVB_ABSOLUTE*100+CPHVB_UINT16:
                    traverse_2<cphvb_uint16, absolute_functor<cphvb_uint16,cphvb_uint16> >( instr );
                    break;
                case CPHVB_ABSOLUTE*100+CPHVB_UINT32:
                    traverse_2<cphvb_uint32, absolute_functor<cphvb_uint32,cphvb_uint32> >( instr );
                    break;
                case CPHVB_ABSOLUTE*100+CPHVB_UINT64:
                    traverse_2<cphvb_uint64, absolute_functor<cphvb_uint64,cphvb_uint64> >( instr );
                    break;
                case CPHVB_ABSOLUTE*100+CPHVB_FLOAT32:
                    traverse_2<cphvb_float32, absolute_functor<cphvb_float32,cphvb_float32> >( instr );
                    break;
                case CPHVB_ABSOLUTE*100+CPHVB_FLOAT64:
                    traverse_2<cphvb_float64, absolute_functor<cphvb_float64,cphvb_float64> >( instr );
                    break;
                case CPHVB_RINT*100+CPHVB_BOOL:
                    traverse_2<cphvb_bool, rint_functor<cphvb_bool,cphvb_bool> >( instr );
                    break;
                case CPHVB_RINT*100+CPHVB_INT8:
                    traverse_2<cphvb_int8, rint_functor<cphvb_int8,cphvb_int8> >( instr );
                    break;
                case CPHVB_RINT*100+CPHVB_INT16:
                    traverse_2<cphvb_int16, rint_functor<cphvb_int16,cphvb_int16> >( instr );
                    break;
                case CPHVB_RINT*100+CPHVB_INT32:
                    traverse_2<cphvb_int32, rint_functor<cphvb_int32,cphvb_int32> >( instr );
                    break;
                case CPHVB_RINT*100+CPHVB_INT64:
                    traverse_2<cphvb_int64, rint_functor<cphvb_int64,cphvb_int64> >( instr );
                    break;
                case CPHVB_RINT*100+CPHVB_UINT8:
                    traverse_2<cphvb_uint8, rint_functor<cphvb_uint8,cphvb_uint8> >( instr );
                    break;
                case CPHVB_RINT*100+CPHVB_UINT16:
                    traverse_2<cphvb_uint16, rint_functor<cphvb_uint16,cphvb_uint16> >( instr );
                    break;
                case CPHVB_RINT*100+CPHVB_UINT32:
                    traverse_2<cphvb_uint32, rint_functor<cphvb_uint32,cphvb_uint32> >( instr );
                    break;
                case CPHVB_RINT*100+CPHVB_UINT64:
                    traverse_2<cphvb_uint64, rint_functor<cphvb_uint64,cphvb_uint64> >( instr );
                    break;
                case CPHVB_RINT*100+CPHVB_FLOAT32:
                    traverse_2<cphvb_float32, rint_functor<cphvb_float32,cphvb_float32> >( instr );
                    break;
                case CPHVB_RINT*100+CPHVB_FLOAT64:
                    traverse_2<cphvb_float64, rint_functor<cphvb_float64,cphvb_float64> >( instr );
                    break;
                case CPHVB_SIGN*100+CPHVB_BOOL:
                    traverse_2<cphvb_bool, sign_functor<cphvb_bool,cphvb_bool> >( instr );
                    break;
                case CPHVB_SIGN*100+CPHVB_INT8:
                    traverse_2<cphvb_int8, sign_functor<cphvb_int8,cphvb_int8> >( instr );
                    break;
                case CPHVB_SIGN*100+CPHVB_INT16:
                    traverse_2<cphvb_int16, sign_functor<cphvb_int16,cphvb_int16> >( instr );
                    break;
                case CPHVB_SIGN*100+CPHVB_INT32:
                    traverse_2<cphvb_int32, sign_functor<cphvb_int32,cphvb_int32> >( instr );
                    break;
                case CPHVB_SIGN*100+CPHVB_INT64:
                    traverse_2<cphvb_int64, sign_functor<cphvb_int64,cphvb_int64> >( instr );
                    break;
                case CPHVB_SIGN*100+CPHVB_UINT8:
                    traverse_2<cphvb_uint8, sign_functor<cphvb_uint8,cphvb_uint8> >( instr );
                    break;
                case CPHVB_SIGN*100+CPHVB_UINT16:
                    traverse_2<cphvb_uint16, sign_functor<cphvb_uint16,cphvb_uint16> >( instr );
                    break;
                case CPHVB_SIGN*100+CPHVB_UINT32:
                    traverse_2<cphvb_uint32, sign_functor<cphvb_uint32,cphvb_uint32> >( instr );
                    break;
                case CPHVB_SIGN*100+CPHVB_UINT64:
                    traverse_2<cphvb_uint64, sign_functor<cphvb_uint64,cphvb_uint64> >( instr );
                    break;
                case CPHVB_SIGN*100+CPHVB_FLOAT32:
                    traverse_2<cphvb_float32, sign_functor<cphvb_float32,cphvb_float32> >( instr );
                    break;
                case CPHVB_SIGN*100+CPHVB_FLOAT64:
                    traverse_2<cphvb_float64, sign_functor<cphvb_float64,cphvb_float64> >( instr );
                    break;
                case CPHVB_CONJ*100+CPHVB_BOOL:
                    traverse_2<cphvb_bool, conj_functor<cphvb_bool,cphvb_bool> >( instr );
                    break;
                case CPHVB_CONJ*100+CPHVB_INT8:
                    traverse_2<cphvb_int8, conj_functor<cphvb_int8,cphvb_int8> >( instr );
                    break;
                case CPHVB_CONJ*100+CPHVB_INT16:
                    traverse_2<cphvb_int16, conj_functor<cphvb_int16,cphvb_int16> >( instr );
                    break;
                case CPHVB_CONJ*100+CPHVB_INT32:
                    traverse_2<cphvb_int32, conj_functor<cphvb_int32,cphvb_int32> >( instr );
                    break;
                case CPHVB_CONJ*100+CPHVB_INT64:
                    traverse_2<cphvb_int64, conj_functor<cphvb_int64,cphvb_int64> >( instr );
                    break;
                case CPHVB_CONJ*100+CPHVB_UINT8:
                    traverse_2<cphvb_uint8, conj_functor<cphvb_uint8,cphvb_uint8> >( instr );
                    break;
                case CPHVB_CONJ*100+CPHVB_UINT16:
                    traverse_2<cphvb_uint16, conj_functor<cphvb_uint16,cphvb_uint16> >( instr );
                    break;
                case CPHVB_CONJ*100+CPHVB_UINT32:
                    traverse_2<cphvb_uint32, conj_functor<cphvb_uint32,cphvb_uint32> >( instr );
                    break;
                case CPHVB_CONJ*100+CPHVB_UINT64:
                    traverse_2<cphvb_uint64, conj_functor<cphvb_uint64,cphvb_uint64> >( instr );
                    break;
                case CPHVB_CONJ*100+CPHVB_FLOAT32:
                    traverse_2<cphvb_float32, conj_functor<cphvb_float32,cphvb_float32> >( instr );
                    break;
                case CPHVB_CONJ*100+CPHVB_FLOAT64:
                    traverse_2<cphvb_float64, conj_functor<cphvb_float64,cphvb_float64> >( instr );
                    break;
                case CPHVB_EXP*100+CPHVB_BOOL:
                    traverse_2<cphvb_bool, exp_functor<cphvb_bool,cphvb_bool> >( instr );
                    break;
                case CPHVB_EXP*100+CPHVB_INT8:
                    traverse_2<cphvb_int8, exp_functor<cphvb_int8,cphvb_int8> >( instr );
                    break;
                case CPHVB_EXP*100+CPHVB_INT16:
                    traverse_2<cphvb_int16, exp_functor<cphvb_int16,cphvb_int16> >( instr );
                    break;
                case CPHVB_EXP*100+CPHVB_INT32:
                    traverse_2<cphvb_int32, exp_functor<cphvb_int32,cphvb_int32> >( instr );
                    break;
                case CPHVB_EXP*100+CPHVB_INT64:
                    traverse_2<cphvb_int64, exp_functor<cphvb_int64,cphvb_int64> >( instr );
                    break;
                case CPHVB_EXP*100+CPHVB_UINT8:
                    traverse_2<cphvb_uint8, exp_functor<cphvb_uint8,cphvb_uint8> >( instr );
                    break;
                case CPHVB_EXP*100+CPHVB_UINT16:
                    traverse_2<cphvb_uint16, exp_functor<cphvb_uint16,cphvb_uint16> >( instr );
                    break;
                case CPHVB_EXP*100+CPHVB_UINT32:
                    traverse_2<cphvb_uint32, exp_functor<cphvb_uint32,cphvb_uint32> >( instr );
                    break;
                case CPHVB_EXP*100+CPHVB_UINT64:
                    traverse_2<cphvb_uint64, exp_functor<cphvb_uint64,cphvb_uint64> >( instr );
                    break;
                case CPHVB_EXP*100+CPHVB_FLOAT32:
                    traverse_2<cphvb_float32, exp_functor<cphvb_float32,cphvb_float32> >( instr );
                    break;
                case CPHVB_EXP*100+CPHVB_FLOAT64:
                    traverse_2<cphvb_float64, exp_functor<cphvb_float64,cphvb_float64> >( instr );
                    break;
                case CPHVB_EXP2*100+CPHVB_BOOL:
                    traverse_2<cphvb_bool, exp2_functor<cphvb_bool,cphvb_bool> >( instr );
                    break;
                case CPHVB_EXP2*100+CPHVB_INT8:
                    traverse_2<cphvb_int8, exp2_functor<cphvb_int8,cphvb_int8> >( instr );
                    break;
                case CPHVB_EXP2*100+CPHVB_INT16:
                    traverse_2<cphvb_int16, exp2_functor<cphvb_int16,cphvb_int16> >( instr );
                    break;
                case CPHVB_EXP2*100+CPHVB_INT32:
                    traverse_2<cphvb_int32, exp2_functor<cphvb_int32,cphvb_int32> >( instr );
                    break;
                case CPHVB_EXP2*100+CPHVB_INT64:
                    traverse_2<cphvb_int64, exp2_functor<cphvb_int64,cphvb_int64> >( instr );
                    break;
                case CPHVB_EXP2*100+CPHVB_UINT8:
                    traverse_2<cphvb_uint8, exp2_functor<cphvb_uint8,cphvb_uint8> >( instr );
                    break;
                case CPHVB_EXP2*100+CPHVB_UINT16:
                    traverse_2<cphvb_uint16, exp2_functor<cphvb_uint16,cphvb_uint16> >( instr );
                    break;
                case CPHVB_EXP2*100+CPHVB_UINT32:
                    traverse_2<cphvb_uint32, exp2_functor<cphvb_uint32,cphvb_uint32> >( instr );
                    break;
                case CPHVB_EXP2*100+CPHVB_UINT64:
                    traverse_2<cphvb_uint64, exp2_functor<cphvb_uint64,cphvb_uint64> >( instr );
                    break;
                case CPHVB_EXP2*100+CPHVB_FLOAT32:
                    traverse_2<cphvb_float32, exp2_functor<cphvb_float32,cphvb_float32> >( instr );
                    break;
                case CPHVB_EXP2*100+CPHVB_FLOAT64:
                    traverse_2<cphvb_float64, exp2_functor<cphvb_float64,cphvb_float64> >( instr );
                    break;
                case CPHVB_LOG2*100+CPHVB_BOOL:
                    traverse_2<cphvb_bool, log2_functor<cphvb_bool,cphvb_bool> >( instr );
                    break;
                case CPHVB_LOG2*100+CPHVB_INT8:
                    traverse_2<cphvb_int8, log2_functor<cphvb_int8,cphvb_int8> >( instr );
                    break;
                case CPHVB_LOG2*100+CPHVB_INT16:
                    traverse_2<cphvb_int16, log2_functor<cphvb_int16,cphvb_int16> >( instr );
                    break;
                case CPHVB_LOG2*100+CPHVB_INT32:
                    traverse_2<cphvb_int32, log2_functor<cphvb_int32,cphvb_int32> >( instr );
                    break;
                case CPHVB_LOG2*100+CPHVB_INT64:
                    traverse_2<cphvb_int64, log2_functor<cphvb_int64,cphvb_int64> >( instr );
                    break;
                case CPHVB_LOG2*100+CPHVB_UINT8:
                    traverse_2<cphvb_uint8, log2_functor<cphvb_uint8,cphvb_uint8> >( instr );
                    break;
                case CPHVB_LOG2*100+CPHVB_UINT16:
                    traverse_2<cphvb_uint16, log2_functor<cphvb_uint16,cphvb_uint16> >( instr );
                    break;
                case CPHVB_LOG2*100+CPHVB_UINT32:
                    traverse_2<cphvb_uint32, log2_functor<cphvb_uint32,cphvb_uint32> >( instr );
                    break;
                case CPHVB_LOG2*100+CPHVB_UINT64:
                    traverse_2<cphvb_uint64, log2_functor<cphvb_uint64,cphvb_uint64> >( instr );
                    break;
                case CPHVB_LOG2*100+CPHVB_FLOAT32:
                    traverse_2<cphvb_float32, log2_functor<cphvb_float32,cphvb_float32> >( instr );
                    break;
                case CPHVB_LOG2*100+CPHVB_FLOAT64:
                    traverse_2<cphvb_float64, log2_functor<cphvb_float64,cphvb_float64> >( instr );
                    break;
                case CPHVB_LOG*100+CPHVB_BOOL:
                    traverse_2<cphvb_bool, log_functor<cphvb_bool,cphvb_bool> >( instr );
                    break;
                case CPHVB_LOG*100+CPHVB_INT8:
                    traverse_2<cphvb_int8, log_functor<cphvb_int8,cphvb_int8> >( instr );
                    break;
                case CPHVB_LOG*100+CPHVB_INT16:
                    traverse_2<cphvb_int16, log_functor<cphvb_int16,cphvb_int16> >( instr );
                    break;
                case CPHVB_LOG*100+CPHVB_INT32:
                    traverse_2<cphvb_int32, log_functor<cphvb_int32,cphvb_int32> >( instr );
                    break;
                case CPHVB_LOG*100+CPHVB_INT64:
                    traverse_2<cphvb_int64, log_functor<cphvb_int64,cphvb_int64> >( instr );
                    break;
                case CPHVB_LOG*100+CPHVB_UINT8:
                    traverse_2<cphvb_uint8, log_functor<cphvb_uint8,cphvb_uint8> >( instr );
                    break;
                case CPHVB_LOG*100+CPHVB_UINT16:
                    traverse_2<cphvb_uint16, log_functor<cphvb_uint16,cphvb_uint16> >( instr );
                    break;
                case CPHVB_LOG*100+CPHVB_UINT32:
                    traverse_2<cphvb_uint32, log_functor<cphvb_uint32,cphvb_uint32> >( instr );
                    break;
                case CPHVB_LOG*100+CPHVB_UINT64:
                    traverse_2<cphvb_uint64, log_functor<cphvb_uint64,cphvb_uint64> >( instr );
                    break;
                case CPHVB_LOG*100+CPHVB_FLOAT32:
                    traverse_2<cphvb_float32, log_functor<cphvb_float32,cphvb_float32> >( instr );
                    break;
                case CPHVB_LOG*100+CPHVB_FLOAT64:
                    traverse_2<cphvb_float64, log_functor<cphvb_float64,cphvb_float64> >( instr );
                    break;
                case CPHVB_LOG10*100+CPHVB_BOOL:
                    traverse_2<cphvb_bool, log10_functor<cphvb_bool,cphvb_bool> >( instr );
                    break;
                case CPHVB_LOG10*100+CPHVB_INT8:
                    traverse_2<cphvb_int8, log10_functor<cphvb_int8,cphvb_int8> >( instr );
                    break;
                case CPHVB_LOG10*100+CPHVB_INT16:
                    traverse_2<cphvb_int16, log10_functor<cphvb_int16,cphvb_int16> >( instr );
                    break;
                case CPHVB_LOG10*100+CPHVB_INT32:
                    traverse_2<cphvb_int32, log10_functor<cphvb_int32,cphvb_int32> >( instr );
                    break;
                case CPHVB_LOG10*100+CPHVB_INT64:
                    traverse_2<cphvb_int64, log10_functor<cphvb_int64,cphvb_int64> >( instr );
                    break;
                case CPHVB_LOG10*100+CPHVB_UINT8:
                    traverse_2<cphvb_uint8, log10_functor<cphvb_uint8,cphvb_uint8> >( instr );
                    break;
                case CPHVB_LOG10*100+CPHVB_UINT16:
                    traverse_2<cphvb_uint16, log10_functor<cphvb_uint16,cphvb_uint16> >( instr );
                    break;
                case CPHVB_LOG10*100+CPHVB_UINT32:
                    traverse_2<cphvb_uint32, log10_functor<cphvb_uint32,cphvb_uint32> >( instr );
                    break;
                case CPHVB_LOG10*100+CPHVB_UINT64:
                    traverse_2<cphvb_uint64, log10_functor<cphvb_uint64,cphvb_uint64> >( instr );
                    break;
                case CPHVB_LOG10*100+CPHVB_FLOAT32:
                    traverse_2<cphvb_float32, log10_functor<cphvb_float32,cphvb_float32> >( instr );
                    break;
                case CPHVB_LOG10*100+CPHVB_FLOAT64:
                    traverse_2<cphvb_float64, log10_functor<cphvb_float64,cphvb_float64> >( instr );
                    break;
                case CPHVB_EXPM1*100+CPHVB_BOOL:
                    traverse_2<cphvb_bool, expm1_functor<cphvb_bool,cphvb_bool> >( instr );
                    break;
                case CPHVB_EXPM1*100+CPHVB_INT8:
                    traverse_2<cphvb_int8, expm1_functor<cphvb_int8,cphvb_int8> >( instr );
                    break;
                case CPHVB_EXPM1*100+CPHVB_INT16:
                    traverse_2<cphvb_int16, expm1_functor<cphvb_int16,cphvb_int16> >( instr );
                    break;
                case CPHVB_EXPM1*100+CPHVB_INT32:
                    traverse_2<cphvb_int32, expm1_functor<cphvb_int32,cphvb_int32> >( instr );
                    break;
                case CPHVB_EXPM1*100+CPHVB_INT64:
                    traverse_2<cphvb_int64, expm1_functor<cphvb_int64,cphvb_int64> >( instr );
                    break;
                case CPHVB_EXPM1*100+CPHVB_UINT8:
                    traverse_2<cphvb_uint8, expm1_functor<cphvb_uint8,cphvb_uint8> >( instr );
                    break;
                case CPHVB_EXPM1*100+CPHVB_UINT16:
                    traverse_2<cphvb_uint16, expm1_functor<cphvb_uint16,cphvb_uint16> >( instr );
                    break;
                case CPHVB_EXPM1*100+CPHVB_UINT32:
                    traverse_2<cphvb_uint32, expm1_functor<cphvb_uint32,cphvb_uint32> >( instr );
                    break;
                case CPHVB_EXPM1*100+CPHVB_UINT64:
                    traverse_2<cphvb_uint64, expm1_functor<cphvb_uint64,cphvb_uint64> >( instr );
                    break;
                case CPHVB_EXPM1*100+CPHVB_FLOAT32:
                    traverse_2<cphvb_float32, expm1_functor<cphvb_float32,cphvb_float32> >( instr );
                    break;
                case CPHVB_EXPM1*100+CPHVB_FLOAT64:
                    traverse_2<cphvb_float64, expm1_functor<cphvb_float64,cphvb_float64> >( instr );
                    break;
                case CPHVB_LOG1P*100+CPHVB_BOOL:
                    traverse_2<cphvb_bool, log1p_functor<cphvb_bool,cphvb_bool> >( instr );
                    break;
                case CPHVB_LOG1P*100+CPHVB_INT8:
                    traverse_2<cphvb_int8, log1p_functor<cphvb_int8,cphvb_int8> >( instr );
                    break;
                case CPHVB_LOG1P*100+CPHVB_INT16:
                    traverse_2<cphvb_int16, log1p_functor<cphvb_int16,cphvb_int16> >( instr );
                    break;
                case CPHVB_LOG1P*100+CPHVB_INT32:
                    traverse_2<cphvb_int32, log1p_functor<cphvb_int32,cphvb_int32> >( instr );
                    break;
                case CPHVB_LOG1P*100+CPHVB_INT64:
                    traverse_2<cphvb_int64, log1p_functor<cphvb_int64,cphvb_int64> >( instr );
                    break;
                case CPHVB_LOG1P*100+CPHVB_UINT8:
                    traverse_2<cphvb_uint8, log1p_functor<cphvb_uint8,cphvb_uint8> >( instr );
                    break;
                case CPHVB_LOG1P*100+CPHVB_UINT16:
                    traverse_2<cphvb_uint16, log1p_functor<cphvb_uint16,cphvb_uint16> >( instr );
                    break;
                case CPHVB_LOG1P*100+CPHVB_UINT32:
                    traverse_2<cphvb_uint32, log1p_functor<cphvb_uint32,cphvb_uint32> >( instr );
                    break;
                case CPHVB_LOG1P*100+CPHVB_UINT64:
                    traverse_2<cphvb_uint64, log1p_functor<cphvb_uint64,cphvb_uint64> >( instr );
                    break;
                case CPHVB_LOG1P*100+CPHVB_FLOAT32:
                    traverse_2<cphvb_float32, log1p_functor<cphvb_float32,cphvb_float32> >( instr );
                    break;
                case CPHVB_LOG1P*100+CPHVB_FLOAT64:
                    traverse_2<cphvb_float64, log1p_functor<cphvb_float64,cphvb_float64> >( instr );
                    break;
                case CPHVB_SQRT*100+CPHVB_BOOL:
                    traverse_2<cphvb_bool, sqrt_functor<cphvb_bool,cphvb_bool> >( instr );
                    break;
                case CPHVB_SQRT*100+CPHVB_INT8:
                    traverse_2<cphvb_int8, sqrt_functor<cphvb_int8,cphvb_int8> >( instr );
                    break;
                case CPHVB_SQRT*100+CPHVB_INT16:
                    traverse_2<cphvb_int16, sqrt_functor<cphvb_int16,cphvb_int16> >( instr );
                    break;
                case CPHVB_SQRT*100+CPHVB_INT32:
                    traverse_2<cphvb_int32, sqrt_functor<cphvb_int32,cphvb_int32> >( instr );
                    break;
                case CPHVB_SQRT*100+CPHVB_INT64:
                    traverse_2<cphvb_int64, sqrt_functor<cphvb_int64,cphvb_int64> >( instr );
                    break;
                case CPHVB_SQRT*100+CPHVB_UINT8:
                    traverse_2<cphvb_uint8, sqrt_functor<cphvb_uint8,cphvb_uint8> >( instr );
                    break;
                case CPHVB_SQRT*100+CPHVB_UINT16:
                    traverse_2<cphvb_uint16, sqrt_functor<cphvb_uint16,cphvb_uint16> >( instr );
                    break;
                case CPHVB_SQRT*100+CPHVB_UINT32:
                    traverse_2<cphvb_uint32, sqrt_functor<cphvb_uint32,cphvb_uint32> >( instr );
                    break;
                case CPHVB_SQRT*100+CPHVB_UINT64:
                    traverse_2<cphvb_uint64, sqrt_functor<cphvb_uint64,cphvb_uint64> >( instr );
                    break;
                case CPHVB_SQRT*100+CPHVB_FLOAT32:
                    traverse_2<cphvb_float32, sqrt_functor<cphvb_float32,cphvb_float32> >( instr );
                    break;
                case CPHVB_SQRT*100+CPHVB_FLOAT64:
                    traverse_2<cphvb_float64, sqrt_functor<cphvb_float64,cphvb_float64> >( instr );
                    break;
                case CPHVB_SQUARE*100+CPHVB_BOOL:
                    traverse_2<cphvb_bool, square_functor<cphvb_bool,cphvb_bool> >( instr );
                    break;
                case CPHVB_SQUARE*100+CPHVB_INT8:
                    traverse_2<cphvb_int8, square_functor<cphvb_int8,cphvb_int8> >( instr );
                    break;
                case CPHVB_SQUARE*100+CPHVB_INT16:
                    traverse_2<cphvb_int16, square_functor<cphvb_int16,cphvb_int16> >( instr );
                    break;
                case CPHVB_SQUARE*100+CPHVB_INT32:
                    traverse_2<cphvb_int32, square_functor<cphvb_int32,cphvb_int32> >( instr );
                    break;
                case CPHVB_SQUARE*100+CPHVB_INT64:
                    traverse_2<cphvb_int64, square_functor<cphvb_int64,cphvb_int64> >( instr );
                    break;
                case CPHVB_SQUARE*100+CPHVB_UINT8:
                    traverse_2<cphvb_uint8, square_functor<cphvb_uint8,cphvb_uint8> >( instr );
                    break;
                case CPHVB_SQUARE*100+CPHVB_UINT16:
                    traverse_2<cphvb_uint16, square_functor<cphvb_uint16,cphvb_uint16> >( instr );
                    break;
                case CPHVB_SQUARE*100+CPHVB_UINT32:
                    traverse_2<cphvb_uint32, square_functor<cphvb_uint32,cphvb_uint32> >( instr );
                    break;
                case CPHVB_SQUARE*100+CPHVB_UINT64:
                    traverse_2<cphvb_uint64, square_functor<cphvb_uint64,cphvb_uint64> >( instr );
                    break;
                case CPHVB_SQUARE*100+CPHVB_FLOAT32:
                    traverse_2<cphvb_float32, square_functor<cphvb_float32,cphvb_float32> >( instr );
                    break;
                case CPHVB_SQUARE*100+CPHVB_FLOAT64:
                    traverse_2<cphvb_float64, square_functor<cphvb_float64,cphvb_float64> >( instr );
                    break;
                case CPHVB_RECIPROCAL*100+CPHVB_BOOL:
                    traverse_2<cphvb_bool, reciprocal_functor<cphvb_bool,cphvb_bool> >( instr );
                    break;
                case CPHVB_RECIPROCAL*100+CPHVB_INT8:
                    traverse_2<cphvb_int8, reciprocal_functor<cphvb_int8,cphvb_int8> >( instr );
                    break;
                case CPHVB_RECIPROCAL*100+CPHVB_INT16:
                    traverse_2<cphvb_int16, reciprocal_functor<cphvb_int16,cphvb_int16> >( instr );
                    break;
                case CPHVB_RECIPROCAL*100+CPHVB_INT32:
                    traverse_2<cphvb_int32, reciprocal_functor<cphvb_int32,cphvb_int32> >( instr );
                    break;
                case CPHVB_RECIPROCAL*100+CPHVB_INT64:
                    traverse_2<cphvb_int64, reciprocal_functor<cphvb_int64,cphvb_int64> >( instr );
                    break;
                case CPHVB_RECIPROCAL*100+CPHVB_UINT8:
                    traverse_2<cphvb_uint8, reciprocal_functor<cphvb_uint8,cphvb_uint8> >( instr );
                    break;
                case CPHVB_RECIPROCAL*100+CPHVB_UINT16:
                    traverse_2<cphvb_uint16, reciprocal_functor<cphvb_uint16,cphvb_uint16> >( instr );
                    break;
                case CPHVB_RECIPROCAL*100+CPHVB_UINT32:
                    traverse_2<cphvb_uint32, reciprocal_functor<cphvb_uint32,cphvb_uint32> >( instr );
                    break;
                case CPHVB_RECIPROCAL*100+CPHVB_UINT64:
                    traverse_2<cphvb_uint64, reciprocal_functor<cphvb_uint64,cphvb_uint64> >( instr );
                    break;
                case CPHVB_RECIPROCAL*100+CPHVB_FLOAT32:
                    traverse_2<cphvb_float32, reciprocal_functor<cphvb_float32,cphvb_float32> >( instr );
                    break;
                case CPHVB_RECIPROCAL*100+CPHVB_FLOAT64:
                    traverse_2<cphvb_float64, reciprocal_functor<cphvb_float64,cphvb_float64> >( instr );
                    break;
                case CPHVB_ONES_LIKE*100+CPHVB_BOOL:
                    traverse_2<cphvb_bool, ones_like_functor<cphvb_bool,cphvb_bool> >( instr );
                    break;
                case CPHVB_ONES_LIKE*100+CPHVB_INT8:
                    traverse_2<cphvb_int8, ones_like_functor<cphvb_int8,cphvb_int8> >( instr );
                    break;
                case CPHVB_ONES_LIKE*100+CPHVB_INT16:
                    traverse_2<cphvb_int16, ones_like_functor<cphvb_int16,cphvb_int16> >( instr );
                    break;
                case CPHVB_ONES_LIKE*100+CPHVB_INT32:
                    traverse_2<cphvb_int32, ones_like_functor<cphvb_int32,cphvb_int32> >( instr );
                    break;
                case CPHVB_ONES_LIKE*100+CPHVB_INT64:
                    traverse_2<cphvb_int64, ones_like_functor<cphvb_int64,cphvb_int64> >( instr );
                    break;
                case CPHVB_ONES_LIKE*100+CPHVB_UINT8:
                    traverse_2<cphvb_uint8, ones_like_functor<cphvb_uint8,cphvb_uint8> >( instr );
                    break;
                case CPHVB_ONES_LIKE*100+CPHVB_UINT16:
                    traverse_2<cphvb_uint16, ones_like_functor<cphvb_uint16,cphvb_uint16> >( instr );
                    break;
                case CPHVB_ONES_LIKE*100+CPHVB_UINT32:
                    traverse_2<cphvb_uint32, ones_like_functor<cphvb_uint32,cphvb_uint32> >( instr );
                    break;
                case CPHVB_ONES_LIKE*100+CPHVB_UINT64:
                    traverse_2<cphvb_uint64, ones_like_functor<cphvb_uint64,cphvb_uint64> >( instr );
                    break;
                case CPHVB_ONES_LIKE*100+CPHVB_FLOAT32:
                    traverse_2<cphvb_float32, ones_like_functor<cphvb_float32,cphvb_float32> >( instr );
                    break;
                case CPHVB_ONES_LIKE*100+CPHVB_FLOAT64:
                    traverse_2<cphvb_float64, ones_like_functor<cphvb_float64,cphvb_float64> >( instr );
                    break;
                case CPHVB_SIN*100+CPHVB_BOOL:
                    traverse_2<cphvb_bool, sin_functor<cphvb_bool,cphvb_bool> >( instr );
                    break;
                case CPHVB_SIN*100+CPHVB_INT8:
                    traverse_2<cphvb_int8, sin_functor<cphvb_int8,cphvb_int8> >( instr );
                    break;
                case CPHVB_SIN*100+CPHVB_INT16:
                    traverse_2<cphvb_int16, sin_functor<cphvb_int16,cphvb_int16> >( instr );
                    break;
                case CPHVB_SIN*100+CPHVB_INT32:
                    traverse_2<cphvb_int32, sin_functor<cphvb_int32,cphvb_int32> >( instr );
                    break;
                case CPHVB_SIN*100+CPHVB_INT64:
                    traverse_2<cphvb_int64, sin_functor<cphvb_int64,cphvb_int64> >( instr );
                    break;
                case CPHVB_SIN*100+CPHVB_UINT8:
                    traverse_2<cphvb_uint8, sin_functor<cphvb_uint8,cphvb_uint8> >( instr );
                    break;
                case CPHVB_SIN*100+CPHVB_UINT16:
                    traverse_2<cphvb_uint16, sin_functor<cphvb_uint16,cphvb_uint16> >( instr );
                    break;
                case CPHVB_SIN*100+CPHVB_UINT32:
                    traverse_2<cphvb_uint32, sin_functor<cphvb_uint32,cphvb_uint32> >( instr );
                    break;
                case CPHVB_SIN*100+CPHVB_UINT64:
                    traverse_2<cphvb_uint64, sin_functor<cphvb_uint64,cphvb_uint64> >( instr );
                    break;
                case CPHVB_SIN*100+CPHVB_FLOAT32:
                    traverse_2<cphvb_float32, sin_functor<cphvb_float32,cphvb_float32> >( instr );
                    break;
                case CPHVB_SIN*100+CPHVB_FLOAT64:
                    traverse_2<cphvb_float64, sin_functor<cphvb_float64,cphvb_float64> >( instr );
                    break;
                case CPHVB_COS*100+CPHVB_BOOL:
                    traverse_2<cphvb_bool, cos_functor<cphvb_bool,cphvb_bool> >( instr );
                    break;
                case CPHVB_COS*100+CPHVB_INT8:
                    traverse_2<cphvb_int8, cos_functor<cphvb_int8,cphvb_int8> >( instr );
                    break;
                case CPHVB_COS*100+CPHVB_INT16:
                    traverse_2<cphvb_int16, cos_functor<cphvb_int16,cphvb_int16> >( instr );
                    break;
                case CPHVB_COS*100+CPHVB_INT32:
                    traverse_2<cphvb_int32, cos_functor<cphvb_int32,cphvb_int32> >( instr );
                    break;
                case CPHVB_COS*100+CPHVB_INT64:
                    traverse_2<cphvb_int64, cos_functor<cphvb_int64,cphvb_int64> >( instr );
                    break;
                case CPHVB_COS*100+CPHVB_UINT8:
                    traverse_2<cphvb_uint8, cos_functor<cphvb_uint8,cphvb_uint8> >( instr );
                    break;
                case CPHVB_COS*100+CPHVB_UINT16:
                    traverse_2<cphvb_uint16, cos_functor<cphvb_uint16,cphvb_uint16> >( instr );
                    break;
                case CPHVB_COS*100+CPHVB_UINT32:
                    traverse_2<cphvb_uint32, cos_functor<cphvb_uint32,cphvb_uint32> >( instr );
                    break;
                case CPHVB_COS*100+CPHVB_UINT64:
                    traverse_2<cphvb_uint64, cos_functor<cphvb_uint64,cphvb_uint64> >( instr );
                    break;
                case CPHVB_COS*100+CPHVB_FLOAT32:
                    traverse_2<cphvb_float32, cos_functor<cphvb_float32,cphvb_float32> >( instr );
                    break;
                case CPHVB_COS*100+CPHVB_FLOAT64:
                    traverse_2<cphvb_float64, cos_functor<cphvb_float64,cphvb_float64> >( instr );
                    break;
                case CPHVB_TAN*100+CPHVB_BOOL:
                    traverse_2<cphvb_bool, tan_functor<cphvb_bool,cphvb_bool> >( instr );
                    break;
                case CPHVB_TAN*100+CPHVB_INT8:
                    traverse_2<cphvb_int8, tan_functor<cphvb_int8,cphvb_int8> >( instr );
                    break;
                case CPHVB_TAN*100+CPHVB_INT16:
                    traverse_2<cphvb_int16, tan_functor<cphvb_int16,cphvb_int16> >( instr );
                    break;
                case CPHVB_TAN*100+CPHVB_INT32:
                    traverse_2<cphvb_int32, tan_functor<cphvb_int32,cphvb_int32> >( instr );
                    break;
                case CPHVB_TAN*100+CPHVB_INT64:
                    traverse_2<cphvb_int64, tan_functor<cphvb_int64,cphvb_int64> >( instr );
                    break;
                case CPHVB_TAN*100+CPHVB_UINT8:
                    traverse_2<cphvb_uint8, tan_functor<cphvb_uint8,cphvb_uint8> >( instr );
                    break;
                case CPHVB_TAN*100+CPHVB_UINT16:
                    traverse_2<cphvb_uint16, tan_functor<cphvb_uint16,cphvb_uint16> >( instr );
                    break;
                case CPHVB_TAN*100+CPHVB_UINT32:
                    traverse_2<cphvb_uint32, tan_functor<cphvb_uint32,cphvb_uint32> >( instr );
                    break;
                case CPHVB_TAN*100+CPHVB_UINT64:
                    traverse_2<cphvb_uint64, tan_functor<cphvb_uint64,cphvb_uint64> >( instr );
                    break;
                case CPHVB_TAN*100+CPHVB_FLOAT32:
                    traverse_2<cphvb_float32, tan_functor<cphvb_float32,cphvb_float32> >( instr );
                    break;
                case CPHVB_TAN*100+CPHVB_FLOAT64:
                    traverse_2<cphvb_float64, tan_functor<cphvb_float64,cphvb_float64> >( instr );
                    break;
                case CPHVB_ARCSIN*100+CPHVB_BOOL:
                    traverse_2<cphvb_bool, arcsin_functor<cphvb_bool,cphvb_bool> >( instr );
                    break;
                case CPHVB_ARCSIN*100+CPHVB_INT8:
                    traverse_2<cphvb_int8, arcsin_functor<cphvb_int8,cphvb_int8> >( instr );
                    break;
                case CPHVB_ARCSIN*100+CPHVB_INT16:
                    traverse_2<cphvb_int16, arcsin_functor<cphvb_int16,cphvb_int16> >( instr );
                    break;
                case CPHVB_ARCSIN*100+CPHVB_INT32:
                    traverse_2<cphvb_int32, arcsin_functor<cphvb_int32,cphvb_int32> >( instr );
                    break;
                case CPHVB_ARCSIN*100+CPHVB_INT64:
                    traverse_2<cphvb_int64, arcsin_functor<cphvb_int64,cphvb_int64> >( instr );
                    break;
                case CPHVB_ARCSIN*100+CPHVB_UINT8:
                    traverse_2<cphvb_uint8, arcsin_functor<cphvb_uint8,cphvb_uint8> >( instr );
                    break;
                case CPHVB_ARCSIN*100+CPHVB_UINT16:
                    traverse_2<cphvb_uint16, arcsin_functor<cphvb_uint16,cphvb_uint16> >( instr );
                    break;
                case CPHVB_ARCSIN*100+CPHVB_UINT32:
                    traverse_2<cphvb_uint32, arcsin_functor<cphvb_uint32,cphvb_uint32> >( instr );
                    break;
                case CPHVB_ARCSIN*100+CPHVB_UINT64:
                    traverse_2<cphvb_uint64, arcsin_functor<cphvb_uint64,cphvb_uint64> >( instr );
                    break;
                case CPHVB_ARCSIN*100+CPHVB_FLOAT32:
                    traverse_2<cphvb_float32, arcsin_functor<cphvb_float32,cphvb_float32> >( instr );
                    break;
                case CPHVB_ARCSIN*100+CPHVB_FLOAT64:
                    traverse_2<cphvb_float64, arcsin_functor<cphvb_float64,cphvb_float64> >( instr );
                    break;
                case CPHVB_ARCCOS*100+CPHVB_BOOL:
                    traverse_2<cphvb_bool, arccos_functor<cphvb_bool,cphvb_bool> >( instr );
                    break;
                case CPHVB_ARCCOS*100+CPHVB_INT8:
                    traverse_2<cphvb_int8, arccos_functor<cphvb_int8,cphvb_int8> >( instr );
                    break;
                case CPHVB_ARCCOS*100+CPHVB_INT16:
                    traverse_2<cphvb_int16, arccos_functor<cphvb_int16,cphvb_int16> >( instr );
                    break;
                case CPHVB_ARCCOS*100+CPHVB_INT32:
                    traverse_2<cphvb_int32, arccos_functor<cphvb_int32,cphvb_int32> >( instr );
                    break;
                case CPHVB_ARCCOS*100+CPHVB_INT64:
                    traverse_2<cphvb_int64, arccos_functor<cphvb_int64,cphvb_int64> >( instr );
                    break;
                case CPHVB_ARCCOS*100+CPHVB_UINT8:
                    traverse_2<cphvb_uint8, arccos_functor<cphvb_uint8,cphvb_uint8> >( instr );
                    break;
                case CPHVB_ARCCOS*100+CPHVB_UINT16:
                    traverse_2<cphvb_uint16, arccos_functor<cphvb_uint16,cphvb_uint16> >( instr );
                    break;
                case CPHVB_ARCCOS*100+CPHVB_UINT32:
                    traverse_2<cphvb_uint32, arccos_functor<cphvb_uint32,cphvb_uint32> >( instr );
                    break;
                case CPHVB_ARCCOS*100+CPHVB_UINT64:
                    traverse_2<cphvb_uint64, arccos_functor<cphvb_uint64,cphvb_uint64> >( instr );
                    break;
                case CPHVB_ARCCOS*100+CPHVB_FLOAT32:
                    traverse_2<cphvb_float32, arccos_functor<cphvb_float32,cphvb_float32> >( instr );
                    break;
                case CPHVB_ARCCOS*100+CPHVB_FLOAT64:
                    traverse_2<cphvb_float64, arccos_functor<cphvb_float64,cphvb_float64> >( instr );
                    break;
                case CPHVB_ARCTAN*100+CPHVB_BOOL:
                    traverse_2<cphvb_bool, arctan_functor<cphvb_bool,cphvb_bool> >( instr );
                    break;
                case CPHVB_ARCTAN*100+CPHVB_INT8:
                    traverse_2<cphvb_int8, arctan_functor<cphvb_int8,cphvb_int8> >( instr );
                    break;
                case CPHVB_ARCTAN*100+CPHVB_INT16:
                    traverse_2<cphvb_int16, arctan_functor<cphvb_int16,cphvb_int16> >( instr );
                    break;
                case CPHVB_ARCTAN*100+CPHVB_INT32:
                    traverse_2<cphvb_int32, arctan_functor<cphvb_int32,cphvb_int32> >( instr );
                    break;
                case CPHVB_ARCTAN*100+CPHVB_INT64:
                    traverse_2<cphvb_int64, arctan_functor<cphvb_int64,cphvb_int64> >( instr );
                    break;
                case CPHVB_ARCTAN*100+CPHVB_UINT8:
                    traverse_2<cphvb_uint8, arctan_functor<cphvb_uint8,cphvb_uint8> >( instr );
                    break;
                case CPHVB_ARCTAN*100+CPHVB_UINT16:
                    traverse_2<cphvb_uint16, arctan_functor<cphvb_uint16,cphvb_uint16> >( instr );
                    break;
                case CPHVB_ARCTAN*100+CPHVB_UINT32:
                    traverse_2<cphvb_uint32, arctan_functor<cphvb_uint32,cphvb_uint32> >( instr );
                    break;
                case CPHVB_ARCTAN*100+CPHVB_UINT64:
                    traverse_2<cphvb_uint64, arctan_functor<cphvb_uint64,cphvb_uint64> >( instr );
                    break;
                case CPHVB_ARCTAN*100+CPHVB_FLOAT32:
                    traverse_2<cphvb_float32, arctan_functor<cphvb_float32,cphvb_float32> >( instr );
                    break;
                case CPHVB_ARCTAN*100+CPHVB_FLOAT64:
                    traverse_2<cphvb_float64, arctan_functor<cphvb_float64,cphvb_float64> >( instr );
                    break;
                case CPHVB_ARCTAN2*100+CPHVB_BOOL:
                    traverse_2<cphvb_bool, arctan2_functor<cphvb_bool,cphvb_bool> >( instr );
                    break;
                case CPHVB_ARCTAN2*100+CPHVB_INT8:
                    traverse_2<cphvb_int8, arctan2_functor<cphvb_int8,cphvb_int8> >( instr );
                    break;
                case CPHVB_ARCTAN2*100+CPHVB_INT16:
                    traverse_2<cphvb_int16, arctan2_functor<cphvb_int16,cphvb_int16> >( instr );
                    break;
                case CPHVB_ARCTAN2*100+CPHVB_INT32:
                    traverse_2<cphvb_int32, arctan2_functor<cphvb_int32,cphvb_int32> >( instr );
                    break;
                case CPHVB_ARCTAN2*100+CPHVB_INT64:
                    traverse_2<cphvb_int64, arctan2_functor<cphvb_int64,cphvb_int64> >( instr );
                    break;
                case CPHVB_ARCTAN2*100+CPHVB_UINT8:
                    traverse_2<cphvb_uint8, arctan2_functor<cphvb_uint8,cphvb_uint8> >( instr );
                    break;
                case CPHVB_ARCTAN2*100+CPHVB_UINT16:
                    traverse_2<cphvb_uint16, arctan2_functor<cphvb_uint16,cphvb_uint16> >( instr );
                    break;
                case CPHVB_ARCTAN2*100+CPHVB_UINT32:
                    traverse_2<cphvb_uint32, arctan2_functor<cphvb_uint32,cphvb_uint32> >( instr );
                    break;
                case CPHVB_ARCTAN2*100+CPHVB_UINT64:
                    traverse_2<cphvb_uint64, arctan2_functor<cphvb_uint64,cphvb_uint64> >( instr );
                    break;
                case CPHVB_ARCTAN2*100+CPHVB_FLOAT32:
                    traverse_2<cphvb_float32, arctan2_functor<cphvb_float32,cphvb_float32> >( instr );
                    break;
                case CPHVB_ARCTAN2*100+CPHVB_FLOAT64:
                    traverse_2<cphvb_float64, arctan2_functor<cphvb_float64,cphvb_float64> >( instr );
                    break;
                case CPHVB_HYPOT*100+CPHVB_BOOL:
                    traverse_2<cphvb_bool, hypot_functor<cphvb_bool,cphvb_bool> >( instr );
                    break;
                case CPHVB_HYPOT*100+CPHVB_INT8:
                    traverse_2<cphvb_int8, hypot_functor<cphvb_int8,cphvb_int8> >( instr );
                    break;
                case CPHVB_HYPOT*100+CPHVB_INT16:
                    traverse_2<cphvb_int16, hypot_functor<cphvb_int16,cphvb_int16> >( instr );
                    break;
                case CPHVB_HYPOT*100+CPHVB_INT32:
                    traverse_2<cphvb_int32, hypot_functor<cphvb_int32,cphvb_int32> >( instr );
                    break;
                case CPHVB_HYPOT*100+CPHVB_INT64:
                    traverse_2<cphvb_int64, hypot_functor<cphvb_int64,cphvb_int64> >( instr );
                    break;
                case CPHVB_HYPOT*100+CPHVB_UINT8:
                    traverse_2<cphvb_uint8, hypot_functor<cphvb_uint8,cphvb_uint8> >( instr );
                    break;
                case CPHVB_HYPOT*100+CPHVB_UINT16:
                    traverse_2<cphvb_uint16, hypot_functor<cphvb_uint16,cphvb_uint16> >( instr );
                    break;
                case CPHVB_HYPOT*100+CPHVB_UINT32:
                    traverse_2<cphvb_uint32, hypot_functor<cphvb_uint32,cphvb_uint32> >( instr );
                    break;
                case CPHVB_HYPOT*100+CPHVB_UINT64:
                    traverse_2<cphvb_uint64, hypot_functor<cphvb_uint64,cphvb_uint64> >( instr );
                    break;
                case CPHVB_HYPOT*100+CPHVB_FLOAT32:
                    traverse_2<cphvb_float32, hypot_functor<cphvb_float32,cphvb_float32> >( instr );
                    break;
                case CPHVB_HYPOT*100+CPHVB_FLOAT64:
                    traverse_2<cphvb_float64, hypot_functor<cphvb_float64,cphvb_float64> >( instr );
                    break;
                case CPHVB_SINH*100+CPHVB_BOOL:
                    traverse_2<cphvb_bool, sinh_functor<cphvb_bool,cphvb_bool> >( instr );
                    break;
                case CPHVB_SINH*100+CPHVB_INT8:
                    traverse_2<cphvb_int8, sinh_functor<cphvb_int8,cphvb_int8> >( instr );
                    break;
                case CPHVB_SINH*100+CPHVB_INT16:
                    traverse_2<cphvb_int16, sinh_functor<cphvb_int16,cphvb_int16> >( instr );
                    break;
                case CPHVB_SINH*100+CPHVB_INT32:
                    traverse_2<cphvb_int32, sinh_functor<cphvb_int32,cphvb_int32> >( instr );
                    break;
                case CPHVB_SINH*100+CPHVB_INT64:
                    traverse_2<cphvb_int64, sinh_functor<cphvb_int64,cphvb_int64> >( instr );
                    break;
                case CPHVB_SINH*100+CPHVB_UINT8:
                    traverse_2<cphvb_uint8, sinh_functor<cphvb_uint8,cphvb_uint8> >( instr );
                    break;
                case CPHVB_SINH*100+CPHVB_UINT16:
                    traverse_2<cphvb_uint16, sinh_functor<cphvb_uint16,cphvb_uint16> >( instr );
                    break;
                case CPHVB_SINH*100+CPHVB_UINT32:
                    traverse_2<cphvb_uint32, sinh_functor<cphvb_uint32,cphvb_uint32> >( instr );
                    break;
                case CPHVB_SINH*100+CPHVB_UINT64:
                    traverse_2<cphvb_uint64, sinh_functor<cphvb_uint64,cphvb_uint64> >( instr );
                    break;
                case CPHVB_SINH*100+CPHVB_FLOAT32:
                    traverse_2<cphvb_float32, sinh_functor<cphvb_float32,cphvb_float32> >( instr );
                    break;
                case CPHVB_SINH*100+CPHVB_FLOAT64:
                    traverse_2<cphvb_float64, sinh_functor<cphvb_float64,cphvb_float64> >( instr );
                    break;
                case CPHVB_COSH*100+CPHVB_BOOL:
                    traverse_2<cphvb_bool, cosh_functor<cphvb_bool,cphvb_bool> >( instr );
                    break;
                case CPHVB_COSH*100+CPHVB_INT8:
                    traverse_2<cphvb_int8, cosh_functor<cphvb_int8,cphvb_int8> >( instr );
                    break;
                case CPHVB_COSH*100+CPHVB_INT16:
                    traverse_2<cphvb_int16, cosh_functor<cphvb_int16,cphvb_int16> >( instr );
                    break;
                case CPHVB_COSH*100+CPHVB_INT32:
                    traverse_2<cphvb_int32, cosh_functor<cphvb_int32,cphvb_int32> >( instr );
                    break;
                case CPHVB_COSH*100+CPHVB_INT64:
                    traverse_2<cphvb_int64, cosh_functor<cphvb_int64,cphvb_int64> >( instr );
                    break;
                case CPHVB_COSH*100+CPHVB_UINT8:
                    traverse_2<cphvb_uint8, cosh_functor<cphvb_uint8,cphvb_uint8> >( instr );
                    break;
                case CPHVB_COSH*100+CPHVB_UINT16:
                    traverse_2<cphvb_uint16, cosh_functor<cphvb_uint16,cphvb_uint16> >( instr );
                    break;
                case CPHVB_COSH*100+CPHVB_UINT32:
                    traverse_2<cphvb_uint32, cosh_functor<cphvb_uint32,cphvb_uint32> >( instr );
                    break;
                case CPHVB_COSH*100+CPHVB_UINT64:
                    traverse_2<cphvb_uint64, cosh_functor<cphvb_uint64,cphvb_uint64> >( instr );
                    break;
                case CPHVB_COSH*100+CPHVB_FLOAT32:
                    traverse_2<cphvb_float32, cosh_functor<cphvb_float32,cphvb_float32> >( instr );
                    break;
                case CPHVB_COSH*100+CPHVB_FLOAT64:
                    traverse_2<cphvb_float64, cosh_functor<cphvb_float64,cphvb_float64> >( instr );
                    break;
                case CPHVB_TANH*100+CPHVB_BOOL:
                    traverse_2<cphvb_bool, tanh_functor<cphvb_bool,cphvb_bool> >( instr );
                    break;
                case CPHVB_TANH*100+CPHVB_INT8:
                    traverse_2<cphvb_int8, tanh_functor<cphvb_int8,cphvb_int8> >( instr );
                    break;
                case CPHVB_TANH*100+CPHVB_INT16:
                    traverse_2<cphvb_int16, tanh_functor<cphvb_int16,cphvb_int16> >( instr );
                    break;
                case CPHVB_TANH*100+CPHVB_INT32:
                    traverse_2<cphvb_int32, tanh_functor<cphvb_int32,cphvb_int32> >( instr );
                    break;
                case CPHVB_TANH*100+CPHVB_INT64:
                    traverse_2<cphvb_int64, tanh_functor<cphvb_int64,cphvb_int64> >( instr );
                    break;
                case CPHVB_TANH*100+CPHVB_UINT8:
                    traverse_2<cphvb_uint8, tanh_functor<cphvb_uint8,cphvb_uint8> >( instr );
                    break;
                case CPHVB_TANH*100+CPHVB_UINT16:
                    traverse_2<cphvb_uint16, tanh_functor<cphvb_uint16,cphvb_uint16> >( instr );
                    break;
                case CPHVB_TANH*100+CPHVB_UINT32:
                    traverse_2<cphvb_uint32, tanh_functor<cphvb_uint32,cphvb_uint32> >( instr );
                    break;
                case CPHVB_TANH*100+CPHVB_UINT64:
                    traverse_2<cphvb_uint64, tanh_functor<cphvb_uint64,cphvb_uint64> >( instr );
                    break;
                case CPHVB_TANH*100+CPHVB_FLOAT32:
                    traverse_2<cphvb_float32, tanh_functor<cphvb_float32,cphvb_float32> >( instr );
                    break;
                case CPHVB_TANH*100+CPHVB_FLOAT64:
                    traverse_2<cphvb_float64, tanh_functor<cphvb_float64,cphvb_float64> >( instr );
                    break;
                case CPHVB_ARCSINH*100+CPHVB_BOOL:
                    traverse_2<cphvb_bool, arcsinh_functor<cphvb_bool,cphvb_bool> >( instr );
                    break;
                case CPHVB_ARCSINH*100+CPHVB_INT8:
                    traverse_2<cphvb_int8, arcsinh_functor<cphvb_int8,cphvb_int8> >( instr );
                    break;
                case CPHVB_ARCSINH*100+CPHVB_INT16:
                    traverse_2<cphvb_int16, arcsinh_functor<cphvb_int16,cphvb_int16> >( instr );
                    break;
                case CPHVB_ARCSINH*100+CPHVB_INT32:
                    traverse_2<cphvb_int32, arcsinh_functor<cphvb_int32,cphvb_int32> >( instr );
                    break;
                case CPHVB_ARCSINH*100+CPHVB_INT64:
                    traverse_2<cphvb_int64, arcsinh_functor<cphvb_int64,cphvb_int64> >( instr );
                    break;
                case CPHVB_ARCSINH*100+CPHVB_UINT8:
                    traverse_2<cphvb_uint8, arcsinh_functor<cphvb_uint8,cphvb_uint8> >( instr );
                    break;
                case CPHVB_ARCSINH*100+CPHVB_UINT16:
                    traverse_2<cphvb_uint16, arcsinh_functor<cphvb_uint16,cphvb_uint16> >( instr );
                    break;
                case CPHVB_ARCSINH*100+CPHVB_UINT32:
                    traverse_2<cphvb_uint32, arcsinh_functor<cphvb_uint32,cphvb_uint32> >( instr );
                    break;
                case CPHVB_ARCSINH*100+CPHVB_UINT64:
                    traverse_2<cphvb_uint64, arcsinh_functor<cphvb_uint64,cphvb_uint64> >( instr );
                    break;
                case CPHVB_ARCSINH*100+CPHVB_FLOAT32:
                    traverse_2<cphvb_float32, arcsinh_functor<cphvb_float32,cphvb_float32> >( instr );
                    break;
                case CPHVB_ARCSINH*100+CPHVB_FLOAT64:
                    traverse_2<cphvb_float64, arcsinh_functor<cphvb_float64,cphvb_float64> >( instr );
                    break;
                case CPHVB_ARCCOSH*100+CPHVB_BOOL:
                    traverse_2<cphvb_bool, arccosh_functor<cphvb_bool,cphvb_bool> >( instr );
                    break;
                case CPHVB_ARCCOSH*100+CPHVB_INT8:
                    traverse_2<cphvb_int8, arccosh_functor<cphvb_int8,cphvb_int8> >( instr );
                    break;
                case CPHVB_ARCCOSH*100+CPHVB_INT16:
                    traverse_2<cphvb_int16, arccosh_functor<cphvb_int16,cphvb_int16> >( instr );
                    break;
                case CPHVB_ARCCOSH*100+CPHVB_INT32:
                    traverse_2<cphvb_int32, arccosh_functor<cphvb_int32,cphvb_int32> >( instr );
                    break;
                case CPHVB_ARCCOSH*100+CPHVB_INT64:
                    traverse_2<cphvb_int64, arccosh_functor<cphvb_int64,cphvb_int64> >( instr );
                    break;
                case CPHVB_ARCCOSH*100+CPHVB_UINT8:
                    traverse_2<cphvb_uint8, arccosh_functor<cphvb_uint8,cphvb_uint8> >( instr );
                    break;
                case CPHVB_ARCCOSH*100+CPHVB_UINT16:
                    traverse_2<cphvb_uint16, arccosh_functor<cphvb_uint16,cphvb_uint16> >( instr );
                    break;
                case CPHVB_ARCCOSH*100+CPHVB_UINT32:
                    traverse_2<cphvb_uint32, arccosh_functor<cphvb_uint32,cphvb_uint32> >( instr );
                    break;
                case CPHVB_ARCCOSH*100+CPHVB_UINT64:
                    traverse_2<cphvb_uint64, arccosh_functor<cphvb_uint64,cphvb_uint64> >( instr );
                    break;
                case CPHVB_ARCCOSH*100+CPHVB_FLOAT32:
                    traverse_2<cphvb_float32, arccosh_functor<cphvb_float32,cphvb_float32> >( instr );
                    break;
                case CPHVB_ARCCOSH*100+CPHVB_FLOAT64:
                    traverse_2<cphvb_float64, arccosh_functor<cphvb_float64,cphvb_float64> >( instr );
                    break;
                case CPHVB_ARCTANH*100+CPHVB_BOOL:
                    traverse_2<cphvb_bool, arctanh_functor<cphvb_bool,cphvb_bool> >( instr );
                    break;
                case CPHVB_ARCTANH*100+CPHVB_INT8:
                    traverse_2<cphvb_int8, arctanh_functor<cphvb_int8,cphvb_int8> >( instr );
                    break;
                case CPHVB_ARCTANH*100+CPHVB_INT16:
                    traverse_2<cphvb_int16, arctanh_functor<cphvb_int16,cphvb_int16> >( instr );
                    break;
                case CPHVB_ARCTANH*100+CPHVB_INT32:
                    traverse_2<cphvb_int32, arctanh_functor<cphvb_int32,cphvb_int32> >( instr );
                    break;
                case CPHVB_ARCTANH*100+CPHVB_INT64:
                    traverse_2<cphvb_int64, arctanh_functor<cphvb_int64,cphvb_int64> >( instr );
                    break;
                case CPHVB_ARCTANH*100+CPHVB_UINT8:
                    traverse_2<cphvb_uint8, arctanh_functor<cphvb_uint8,cphvb_uint8> >( instr );
                    break;
                case CPHVB_ARCTANH*100+CPHVB_UINT16:
                    traverse_2<cphvb_uint16, arctanh_functor<cphvb_uint16,cphvb_uint16> >( instr );
                    break;
                case CPHVB_ARCTANH*100+CPHVB_UINT32:
                    traverse_2<cphvb_uint32, arctanh_functor<cphvb_uint32,cphvb_uint32> >( instr );
                    break;
                case CPHVB_ARCTANH*100+CPHVB_UINT64:
                    traverse_2<cphvb_uint64, arctanh_functor<cphvb_uint64,cphvb_uint64> >( instr );
                    break;
                case CPHVB_ARCTANH*100+CPHVB_FLOAT32:
                    traverse_2<cphvb_float32, arctanh_functor<cphvb_float32,cphvb_float32> >( instr );
                    break;
                case CPHVB_ARCTANH*100+CPHVB_FLOAT64:
                    traverse_2<cphvb_float64, arctanh_functor<cphvb_float64,cphvb_float64> >( instr );
                    break;
                case CPHVB_DEG2RAD*100+CPHVB_BOOL:
                    traverse_2<cphvb_bool, deg2rad_functor<cphvb_bool,cphvb_bool> >( instr );
                    break;
                case CPHVB_DEG2RAD*100+CPHVB_INT8:
                    traverse_2<cphvb_int8, deg2rad_functor<cphvb_int8,cphvb_int8> >( instr );
                    break;
                case CPHVB_DEG2RAD*100+CPHVB_INT16:
                    traverse_2<cphvb_int16, deg2rad_functor<cphvb_int16,cphvb_int16> >( instr );
                    break;
                case CPHVB_DEG2RAD*100+CPHVB_INT32:
                    traverse_2<cphvb_int32, deg2rad_functor<cphvb_int32,cphvb_int32> >( instr );
                    break;
                case CPHVB_DEG2RAD*100+CPHVB_INT64:
                    traverse_2<cphvb_int64, deg2rad_functor<cphvb_int64,cphvb_int64> >( instr );
                    break;
                case CPHVB_DEG2RAD*100+CPHVB_UINT8:
                    traverse_2<cphvb_uint8, deg2rad_functor<cphvb_uint8,cphvb_uint8> >( instr );
                    break;
                case CPHVB_DEG2RAD*100+CPHVB_UINT16:
                    traverse_2<cphvb_uint16, deg2rad_functor<cphvb_uint16,cphvb_uint16> >( instr );
                    break;
                case CPHVB_DEG2RAD*100+CPHVB_UINT32:
                    traverse_2<cphvb_uint32, deg2rad_functor<cphvb_uint32,cphvb_uint32> >( instr );
                    break;
                case CPHVB_DEG2RAD*100+CPHVB_UINT64:
                    traverse_2<cphvb_uint64, deg2rad_functor<cphvb_uint64,cphvb_uint64> >( instr );
                    break;
                case CPHVB_DEG2RAD*100+CPHVB_FLOAT32:
                    traverse_2<cphvb_float32, deg2rad_functor<cphvb_float32,cphvb_float32> >( instr );
                    break;
                case CPHVB_DEG2RAD*100+CPHVB_FLOAT64:
                    traverse_2<cphvb_float64, deg2rad_functor<cphvb_float64,cphvb_float64> >( instr );
                    break;
                case CPHVB_RAD2DEG*100+CPHVB_BOOL:
                    traverse_2<cphvb_bool, rad2deg_functor<cphvb_bool,cphvb_bool> >( instr );
                    break;
                case CPHVB_RAD2DEG*100+CPHVB_INT8:
                    traverse_2<cphvb_int8, rad2deg_functor<cphvb_int8,cphvb_int8> >( instr );
                    break;
                case CPHVB_RAD2DEG*100+CPHVB_INT16:
                    traverse_2<cphvb_int16, rad2deg_functor<cphvb_int16,cphvb_int16> >( instr );
                    break;
                case CPHVB_RAD2DEG*100+CPHVB_INT32:
                    traverse_2<cphvb_int32, rad2deg_functor<cphvb_int32,cphvb_int32> >( instr );
                    break;
                case CPHVB_RAD2DEG*100+CPHVB_INT64:
                    traverse_2<cphvb_int64, rad2deg_functor<cphvb_int64,cphvb_int64> >( instr );
                    break;
                case CPHVB_RAD2DEG*100+CPHVB_UINT8:
                    traverse_2<cphvb_uint8, rad2deg_functor<cphvb_uint8,cphvb_uint8> >( instr );
                    break;
                case CPHVB_RAD2DEG*100+CPHVB_UINT16:
                    traverse_2<cphvb_uint16, rad2deg_functor<cphvb_uint16,cphvb_uint16> >( instr );
                    break;
                case CPHVB_RAD2DEG*100+CPHVB_UINT32:
                    traverse_2<cphvb_uint32, rad2deg_functor<cphvb_uint32,cphvb_uint32> >( instr );
                    break;
                case CPHVB_RAD2DEG*100+CPHVB_UINT64:
                    traverse_2<cphvb_uint64, rad2deg_functor<cphvb_uint64,cphvb_uint64> >( instr );
                    break;
                case CPHVB_RAD2DEG*100+CPHVB_FLOAT32:
                    traverse_2<cphvb_float32, rad2deg_functor<cphvb_float32,cphvb_float32> >( instr );
                    break;
                case CPHVB_RAD2DEG*100+CPHVB_FLOAT64:
                    traverse_2<cphvb_float64, rad2deg_functor<cphvb_float64,cphvb_float64> >( instr );
                    break;
                case CPHVB_LOGICAL_NOT*100+CPHVB_BOOL:
                    traverse_2<cphvb_bool, logical_not_functor<cphvb_bool,cphvb_bool> >( instr );
                    break;
                case CPHVB_LOGICAL_NOT*100+CPHVB_INT8:
                    traverse_2<cphvb_int8, logical_not_functor<cphvb_int8,cphvb_int8> >( instr );
                    break;
                case CPHVB_LOGICAL_NOT*100+CPHVB_INT16:
                    traverse_2<cphvb_int16, logical_not_functor<cphvb_int16,cphvb_int16> >( instr );
                    break;
                case CPHVB_LOGICAL_NOT*100+CPHVB_INT32:
                    traverse_2<cphvb_int32, logical_not_functor<cphvb_int32,cphvb_int32> >( instr );
                    break;
                case CPHVB_LOGICAL_NOT*100+CPHVB_INT64:
                    traverse_2<cphvb_int64, logical_not_functor<cphvb_int64,cphvb_int64> >( instr );
                    break;
                case CPHVB_LOGICAL_NOT*100+CPHVB_UINT8:
                    traverse_2<cphvb_uint8, logical_not_functor<cphvb_uint8,cphvb_uint8> >( instr );
                    break;
                case CPHVB_LOGICAL_NOT*100+CPHVB_UINT16:
                    traverse_2<cphvb_uint16, logical_not_functor<cphvb_uint16,cphvb_uint16> >( instr );
                    break;
                case CPHVB_LOGICAL_NOT*100+CPHVB_UINT32:
                    traverse_2<cphvb_uint32, logical_not_functor<cphvb_uint32,cphvb_uint32> >( instr );
                    break;
                case CPHVB_LOGICAL_NOT*100+CPHVB_UINT64:
                    traverse_2<cphvb_uint64, logical_not_functor<cphvb_uint64,cphvb_uint64> >( instr );
                    break;
                case CPHVB_LOGICAL_NOT*100+CPHVB_FLOAT32:
                    traverse_2<cphvb_float32, logical_not_functor<cphvb_float32,cphvb_float32> >( instr );
                    break;
                case CPHVB_LOGICAL_NOT*100+CPHVB_FLOAT64:
                    traverse_2<cphvb_float64, logical_not_functor<cphvb_float64,cphvb_float64> >( instr );
                    break;
                case CPHVB_INVERT*100+CPHVB_BOOL:
                    traverse_2<cphvb_bool, invert_functor<cphvb_bool,cphvb_bool> >( instr );
                    break;
                case CPHVB_INVERT*100+CPHVB_INT8:
                    traverse_2<cphvb_int8, invert_functor<cphvb_int8,cphvb_int8> >( instr );
                    break;
                case CPHVB_INVERT*100+CPHVB_INT16:
                    traverse_2<cphvb_int16, invert_functor<cphvb_int16,cphvb_int16> >( instr );
                    break;
                case CPHVB_INVERT*100+CPHVB_INT32:
                    traverse_2<cphvb_int32, invert_functor<cphvb_int32,cphvb_int32> >( instr );
                    break;
                case CPHVB_INVERT*100+CPHVB_INT64:
                    traverse_2<cphvb_int64, invert_functor<cphvb_int64,cphvb_int64> >( instr );
                    break;
                case CPHVB_INVERT*100+CPHVB_UINT8:
                    traverse_2<cphvb_uint8, invert_functor<cphvb_uint8,cphvb_uint8> >( instr );
                    break;
                case CPHVB_INVERT*100+CPHVB_UINT16:
                    traverse_2<cphvb_uint16, invert_functor<cphvb_uint16,cphvb_uint16> >( instr );
                    break;
                case CPHVB_INVERT*100+CPHVB_UINT32:
                    traverse_2<cphvb_uint32, invert_functor<cphvb_uint32,cphvb_uint32> >( instr );
                    break;
                case CPHVB_INVERT*100+CPHVB_UINT64:
                    traverse_2<cphvb_uint64, invert_functor<cphvb_uint64,cphvb_uint64> >( instr );
                    break;
                case CPHVB_INVERT*100+CPHVB_FLOAT32:
                    traverse_2<cphvb_float32, invert_functor<cphvb_float32,cphvb_float32> >( instr );
                    break;
                case CPHVB_INVERT*100+CPHVB_FLOAT64:
                    traverse_2<cphvb_float64, invert_functor<cphvb_float64,cphvb_float64> >( instr );
                    break;
                case CPHVB_ISFINITE*100+CPHVB_BOOL:
                    traverse_2<cphvb_bool, isfinite_functor<cphvb_bool,cphvb_bool> >( instr );
                    break;
                case CPHVB_ISFINITE*100+CPHVB_INT8:
                    traverse_2<cphvb_int8, isfinite_functor<cphvb_int8,cphvb_int8> >( instr );
                    break;
                case CPHVB_ISFINITE*100+CPHVB_INT16:
                    traverse_2<cphvb_int16, isfinite_functor<cphvb_int16,cphvb_int16> >( instr );
                    break;
                case CPHVB_ISFINITE*100+CPHVB_INT32:
                    traverse_2<cphvb_int32, isfinite_functor<cphvb_int32,cphvb_int32> >( instr );
                    break;
                case CPHVB_ISFINITE*100+CPHVB_INT64:
                    traverse_2<cphvb_int64, isfinite_functor<cphvb_int64,cphvb_int64> >( instr );
                    break;
                case CPHVB_ISFINITE*100+CPHVB_UINT8:
                    traverse_2<cphvb_uint8, isfinite_functor<cphvb_uint8,cphvb_uint8> >( instr );
                    break;
                case CPHVB_ISFINITE*100+CPHVB_UINT16:
                    traverse_2<cphvb_uint16, isfinite_functor<cphvb_uint16,cphvb_uint16> >( instr );
                    break;
                case CPHVB_ISFINITE*100+CPHVB_UINT32:
                    traverse_2<cphvb_uint32, isfinite_functor<cphvb_uint32,cphvb_uint32> >( instr );
                    break;
                case CPHVB_ISFINITE*100+CPHVB_UINT64:
                    traverse_2<cphvb_uint64, isfinite_functor<cphvb_uint64,cphvb_uint64> >( instr );
                    break;
                case CPHVB_ISFINITE*100+CPHVB_FLOAT32:
                    traverse_2<cphvb_float32, isfinite_functor<cphvb_float32,cphvb_float32> >( instr );
                    break;
                case CPHVB_ISFINITE*100+CPHVB_FLOAT64:
                    traverse_2<cphvb_float64, isfinite_functor<cphvb_float64,cphvb_float64> >( instr );
                    break;
                case CPHVB_ISINF*100+CPHVB_BOOL:
                    traverse_2<cphvb_bool, isinf_functor<cphvb_bool,cphvb_bool> >( instr );
                    break;
                case CPHVB_ISINF*100+CPHVB_INT8:
                    traverse_2<cphvb_int8, isinf_functor<cphvb_int8,cphvb_int8> >( instr );
                    break;
                case CPHVB_ISINF*100+CPHVB_INT16:
                    traverse_2<cphvb_int16, isinf_functor<cphvb_int16,cphvb_int16> >( instr );
                    break;
                case CPHVB_ISINF*100+CPHVB_INT32:
                    traverse_2<cphvb_int32, isinf_functor<cphvb_int32,cphvb_int32> >( instr );
                    break;
                case CPHVB_ISINF*100+CPHVB_INT64:
                    traverse_2<cphvb_int64, isinf_functor<cphvb_int64,cphvb_int64> >( instr );
                    break;
                case CPHVB_ISINF*100+CPHVB_UINT8:
                    traverse_2<cphvb_uint8, isinf_functor<cphvb_uint8,cphvb_uint8> >( instr );
                    break;
                case CPHVB_ISINF*100+CPHVB_UINT16:
                    traverse_2<cphvb_uint16, isinf_functor<cphvb_uint16,cphvb_uint16> >( instr );
                    break;
                case CPHVB_ISINF*100+CPHVB_UINT32:
                    traverse_2<cphvb_uint32, isinf_functor<cphvb_uint32,cphvb_uint32> >( instr );
                    break;
                case CPHVB_ISINF*100+CPHVB_UINT64:
                    traverse_2<cphvb_uint64, isinf_functor<cphvb_uint64,cphvb_uint64> >( instr );
                    break;
                case CPHVB_ISINF*100+CPHVB_FLOAT32:
                    traverse_2<cphvb_float32, isinf_functor<cphvb_float32,cphvb_float32> >( instr );
                    break;
                case CPHVB_ISINF*100+CPHVB_FLOAT64:
                    traverse_2<cphvb_float64, isinf_functor<cphvb_float64,cphvb_float64> >( instr );
                    break;
                case CPHVB_ISNAN*100+CPHVB_BOOL:
                    traverse_2<cphvb_bool, isnan_functor<cphvb_bool,cphvb_bool> >( instr );
                    break;
                case CPHVB_ISNAN*100+CPHVB_INT8:
                    traverse_2<cphvb_int8, isnan_functor<cphvb_int8,cphvb_int8> >( instr );
                    break;
                case CPHVB_ISNAN*100+CPHVB_INT16:
                    traverse_2<cphvb_int16, isnan_functor<cphvb_int16,cphvb_int16> >( instr );
                    break;
                case CPHVB_ISNAN*100+CPHVB_INT32:
                    traverse_2<cphvb_int32, isnan_functor<cphvb_int32,cphvb_int32> >( instr );
                    break;
                case CPHVB_ISNAN*100+CPHVB_INT64:
                    traverse_2<cphvb_int64, isnan_functor<cphvb_int64,cphvb_int64> >( instr );
                    break;
                case CPHVB_ISNAN*100+CPHVB_UINT8:
                    traverse_2<cphvb_uint8, isnan_functor<cphvb_uint8,cphvb_uint8> >( instr );
                    break;
                case CPHVB_ISNAN*100+CPHVB_UINT16:
                    traverse_2<cphvb_uint16, isnan_functor<cphvb_uint16,cphvb_uint16> >( instr );
                    break;
                case CPHVB_ISNAN*100+CPHVB_UINT32:
                    traverse_2<cphvb_uint32, isnan_functor<cphvb_uint32,cphvb_uint32> >( instr );
                    break;
                case CPHVB_ISNAN*100+CPHVB_UINT64:
                    traverse_2<cphvb_uint64, isnan_functor<cphvb_uint64,cphvb_uint64> >( instr );
                    break;
                case CPHVB_ISNAN*100+CPHVB_FLOAT32:
                    traverse_2<cphvb_float32, isnan_functor<cphvb_float32,cphvb_float32> >( instr );
                    break;
                case CPHVB_ISNAN*100+CPHVB_FLOAT64:
                    traverse_2<cphvb_float64, isnan_functor<cphvb_float64,cphvb_float64> >( instr );
                    break;
                case CPHVB_SIGNBIT*100+CPHVB_BOOL:
                    traverse_2<cphvb_bool, signbit_functor<cphvb_bool,cphvb_bool> >( instr );
                    break;
                case CPHVB_SIGNBIT*100+CPHVB_INT8:
                    traverse_2<cphvb_int8, signbit_functor<cphvb_int8,cphvb_int8> >( instr );
                    break;
                case CPHVB_SIGNBIT*100+CPHVB_INT16:
                    traverse_2<cphvb_int16, signbit_functor<cphvb_int16,cphvb_int16> >( instr );
                    break;
                case CPHVB_SIGNBIT*100+CPHVB_INT32:
                    traverse_2<cphvb_int32, signbit_functor<cphvb_int32,cphvb_int32> >( instr );
                    break;
                case CPHVB_SIGNBIT*100+CPHVB_INT64:
                    traverse_2<cphvb_int64, signbit_functor<cphvb_int64,cphvb_int64> >( instr );
                    break;
                case CPHVB_SIGNBIT*100+CPHVB_UINT8:
                    traverse_2<cphvb_uint8, signbit_functor<cphvb_uint8,cphvb_uint8> >( instr );
                    break;
                case CPHVB_SIGNBIT*100+CPHVB_UINT16:
                    traverse_2<cphvb_uint16, signbit_functor<cphvb_uint16,cphvb_uint16> >( instr );
                    break;
                case CPHVB_SIGNBIT*100+CPHVB_UINT32:
                    traverse_2<cphvb_uint32, signbit_functor<cphvb_uint32,cphvb_uint32> >( instr );
                    break;
                case CPHVB_SIGNBIT*100+CPHVB_UINT64:
                    traverse_2<cphvb_uint64, signbit_functor<cphvb_uint64,cphvb_uint64> >( instr );
                    break;
                case CPHVB_SIGNBIT*100+CPHVB_FLOAT32:
                    traverse_2<cphvb_float32, signbit_functor<cphvb_float32,cphvb_float32> >( instr );
                    break;
                case CPHVB_SIGNBIT*100+CPHVB_FLOAT64:
                    traverse_2<cphvb_float64, signbit_functor<cphvb_float64,cphvb_float64> >( instr );
                    break;
                case CPHVB_FLOOR*100+CPHVB_BOOL:
                    traverse_2<cphvb_bool, floor_functor<cphvb_bool,cphvb_bool> >( instr );
                    break;
                case CPHVB_FLOOR*100+CPHVB_INT8:
                    traverse_2<cphvb_int8, floor_functor<cphvb_int8,cphvb_int8> >( instr );
                    break;
                case CPHVB_FLOOR*100+CPHVB_INT16:
                    traverse_2<cphvb_int16, floor_functor<cphvb_int16,cphvb_int16> >( instr );
                    break;
                case CPHVB_FLOOR*100+CPHVB_INT32:
                    traverse_2<cphvb_int32, floor_functor<cphvb_int32,cphvb_int32> >( instr );
                    break;
                case CPHVB_FLOOR*100+CPHVB_INT64:
                    traverse_2<cphvb_int64, floor_functor<cphvb_int64,cphvb_int64> >( instr );
                    break;
                case CPHVB_FLOOR*100+CPHVB_UINT8:
                    traverse_2<cphvb_uint8, floor_functor<cphvb_uint8,cphvb_uint8> >( instr );
                    break;
                case CPHVB_FLOOR*100+CPHVB_UINT16:
                    traverse_2<cphvb_uint16, floor_functor<cphvb_uint16,cphvb_uint16> >( instr );
                    break;
                case CPHVB_FLOOR*100+CPHVB_UINT32:
                    traverse_2<cphvb_uint32, floor_functor<cphvb_uint32,cphvb_uint32> >( instr );
                    break;
                case CPHVB_FLOOR*100+CPHVB_UINT64:
                    traverse_2<cphvb_uint64, floor_functor<cphvb_uint64,cphvb_uint64> >( instr );
                    break;
                case CPHVB_FLOOR*100+CPHVB_FLOAT32:
                    traverse_2<cphvb_float32, floor_functor<cphvb_float32,cphvb_float32> >( instr );
                    break;
                case CPHVB_FLOOR*100+CPHVB_FLOAT64:
                    traverse_2<cphvb_float64, floor_functor<cphvb_float64,cphvb_float64> >( instr );
                    break;
                case CPHVB_CEIL*100+CPHVB_BOOL:
                    traverse_2<cphvb_bool, ceil_functor<cphvb_bool,cphvb_bool> >( instr );
                    break;
                case CPHVB_CEIL*100+CPHVB_INT8:
                    traverse_2<cphvb_int8, ceil_functor<cphvb_int8,cphvb_int8> >( instr );
                    break;
                case CPHVB_CEIL*100+CPHVB_INT16:
                    traverse_2<cphvb_int16, ceil_functor<cphvb_int16,cphvb_int16> >( instr );
                    break;
                case CPHVB_CEIL*100+CPHVB_INT32:
                    traverse_2<cphvb_int32, ceil_functor<cphvb_int32,cphvb_int32> >( instr );
                    break;
                case CPHVB_CEIL*100+CPHVB_INT64:
                    traverse_2<cphvb_int64, ceil_functor<cphvb_int64,cphvb_int64> >( instr );
                    break;
                case CPHVB_CEIL*100+CPHVB_UINT8:
                    traverse_2<cphvb_uint8, ceil_functor<cphvb_uint8,cphvb_uint8> >( instr );
                    break;
                case CPHVB_CEIL*100+CPHVB_UINT16:
                    traverse_2<cphvb_uint16, ceil_functor<cphvb_uint16,cphvb_uint16> >( instr );
                    break;
                case CPHVB_CEIL*100+CPHVB_UINT32:
                    traverse_2<cphvb_uint32, ceil_functor<cphvb_uint32,cphvb_uint32> >( instr );
                    break;
                case CPHVB_CEIL*100+CPHVB_UINT64:
                    traverse_2<cphvb_uint64, ceil_functor<cphvb_uint64,cphvb_uint64> >( instr );
                    break;
                case CPHVB_CEIL*100+CPHVB_FLOAT32:
                    traverse_2<cphvb_float32, ceil_functor<cphvb_float32,cphvb_float32> >( instr );
                    break;
                case CPHVB_CEIL*100+CPHVB_FLOAT64:
                    traverse_2<cphvb_float64, ceil_functor<cphvb_float64,cphvb_float64> >( instr );
                    break;
                case CPHVB_TRUNC*100+CPHVB_BOOL:
                    traverse_2<cphvb_bool, trunc_functor<cphvb_bool,cphvb_bool> >( instr );
                    break;
                case CPHVB_TRUNC*100+CPHVB_INT8:
                    traverse_2<cphvb_int8, trunc_functor<cphvb_int8,cphvb_int8> >( instr );
                    break;
                case CPHVB_TRUNC*100+CPHVB_INT16:
                    traverse_2<cphvb_int16, trunc_functor<cphvb_int16,cphvb_int16> >( instr );
                    break;
                case CPHVB_TRUNC*100+CPHVB_INT32:
                    traverse_2<cphvb_int32, trunc_functor<cphvb_int32,cphvb_int32> >( instr );
                    break;
                case CPHVB_TRUNC*100+CPHVB_INT64:
                    traverse_2<cphvb_int64, trunc_functor<cphvb_int64,cphvb_int64> >( instr );
                    break;
                case CPHVB_TRUNC*100+CPHVB_UINT8:
                    traverse_2<cphvb_uint8, trunc_functor<cphvb_uint8,cphvb_uint8> >( instr );
                    break;
                case CPHVB_TRUNC*100+CPHVB_UINT16:
                    traverse_2<cphvb_uint16, trunc_functor<cphvb_uint16,cphvb_uint16> >( instr );
                    break;
                case CPHVB_TRUNC*100+CPHVB_UINT32:
                    traverse_2<cphvb_uint32, trunc_functor<cphvb_uint32,cphvb_uint32> >( instr );
                    break;
                case CPHVB_TRUNC*100+CPHVB_UINT64:
                    traverse_2<cphvb_uint64, trunc_functor<cphvb_uint64,cphvb_uint64> >( instr );
                    break;
                case CPHVB_TRUNC*100+CPHVB_FLOAT32:
                    traverse_2<cphvb_float32, trunc_functor<cphvb_float32,cphvb_float32> >( instr );
                    break;
                case CPHVB_TRUNC*100+CPHVB_FLOAT64:
                    traverse_2<cphvb_float64, trunc_functor<cphvb_float64,cphvb_float64> >( instr );
                    break;
                case CPHVB_ISREAL*100+CPHVB_BOOL:
                    traverse_2<cphvb_bool, isreal_functor<cphvb_bool,cphvb_bool> >( instr );
                    break;
                case CPHVB_ISREAL*100+CPHVB_INT8:
                    traverse_2<cphvb_int8, isreal_functor<cphvb_int8,cphvb_int8> >( instr );
                    break;
                case CPHVB_ISREAL*100+CPHVB_INT16:
                    traverse_2<cphvb_int16, isreal_functor<cphvb_int16,cphvb_int16> >( instr );
                    break;
                case CPHVB_ISREAL*100+CPHVB_INT32:
                    traverse_2<cphvb_int32, isreal_functor<cphvb_int32,cphvb_int32> >( instr );
                    break;
                case CPHVB_ISREAL*100+CPHVB_INT64:
                    traverse_2<cphvb_int64, isreal_functor<cphvb_int64,cphvb_int64> >( instr );
                    break;
                case CPHVB_ISREAL*100+CPHVB_UINT8:
                    traverse_2<cphvb_uint8, isreal_functor<cphvb_uint8,cphvb_uint8> >( instr );
                    break;
                case CPHVB_ISREAL*100+CPHVB_UINT16:
                    traverse_2<cphvb_uint16, isreal_functor<cphvb_uint16,cphvb_uint16> >( instr );
                    break;
                case CPHVB_ISREAL*100+CPHVB_UINT32:
                    traverse_2<cphvb_uint32, isreal_functor<cphvb_uint32,cphvb_uint32> >( instr );
                    break;
                case CPHVB_ISREAL*100+CPHVB_UINT64:
                    traverse_2<cphvb_uint64, isreal_functor<cphvb_uint64,cphvb_uint64> >( instr );
                    break;
                case CPHVB_ISREAL*100+CPHVB_FLOAT32:
                    traverse_2<cphvb_float32, isreal_functor<cphvb_float32,cphvb_float32> >( instr );
                    break;
                case CPHVB_ISREAL*100+CPHVB_FLOAT64:
                    traverse_2<cphvb_float64, isreal_functor<cphvb_float64,cphvb_float64> >( instr );
                    break;
                case CPHVB_ISCOMPLEX*100+CPHVB_BOOL:
                    traverse_2<cphvb_bool, iscomplex_functor<cphvb_bool,cphvb_bool> >( instr );
                    break;
                case CPHVB_ISCOMPLEX*100+CPHVB_INT8:
                    traverse_2<cphvb_int8, iscomplex_functor<cphvb_int8,cphvb_int8> >( instr );
                    break;
                case CPHVB_ISCOMPLEX*100+CPHVB_INT16:
                    traverse_2<cphvb_int16, iscomplex_functor<cphvb_int16,cphvb_int16> >( instr );
                    break;
                case CPHVB_ISCOMPLEX*100+CPHVB_INT32:
                    traverse_2<cphvb_int32, iscomplex_functor<cphvb_int32,cphvb_int32> >( instr );
                    break;
                case CPHVB_ISCOMPLEX*100+CPHVB_INT64:
                    traverse_2<cphvb_int64, iscomplex_functor<cphvb_int64,cphvb_int64> >( instr );
                    break;
                case CPHVB_ISCOMPLEX*100+CPHVB_UINT8:
                    traverse_2<cphvb_uint8, iscomplex_functor<cphvb_uint8,cphvb_uint8> >( instr );
                    break;
                case CPHVB_ISCOMPLEX*100+CPHVB_UINT16:
                    traverse_2<cphvb_uint16, iscomplex_functor<cphvb_uint16,cphvb_uint16> >( instr );
                    break;
                case CPHVB_ISCOMPLEX*100+CPHVB_UINT32:
                    traverse_2<cphvb_uint32, iscomplex_functor<cphvb_uint32,cphvb_uint32> >( instr );
                    break;
                case CPHVB_ISCOMPLEX*100+CPHVB_UINT64:
                    traverse_2<cphvb_uint64, iscomplex_functor<cphvb_uint64,cphvb_uint64> >( instr );
                    break;
                case CPHVB_ISCOMPLEX*100+CPHVB_FLOAT32:
                    traverse_2<cphvb_float32, iscomplex_functor<cphvb_float32,cphvb_float32> >( instr );
                    break;
                case CPHVB_ISCOMPLEX*100+CPHVB_FLOAT64:
                    traverse_2<cphvb_float64, iscomplex_functor<cphvb_float64,cphvb_float64> >( instr );
                    break;
                case CPHVB_MODF*100+CPHVB_BOOL:
                    traverse_3<cphvb_bool, modf_functor<cphvb_bool,cphvb_bool,cphvb_bool> >( instr );
                    break;
                case CPHVB_MODF*100+CPHVB_INT8:
                    traverse_3<cphvb_int8, modf_functor<cphvb_int8,cphvb_int8,cphvb_int8> >( instr );
                    break;
                case CPHVB_MODF*100+CPHVB_INT16:
                    traverse_3<cphvb_int16, modf_functor<cphvb_int16,cphvb_int16,cphvb_int16> >( instr );
                    break;
                case CPHVB_MODF*100+CPHVB_INT32:
                    traverse_3<cphvb_int32, modf_functor<cphvb_int32,cphvb_int32,cphvb_int32> >( instr );
                    break;
                case CPHVB_MODF*100+CPHVB_INT64:
                    traverse_3<cphvb_int64, modf_functor<cphvb_int64,cphvb_int64,cphvb_int64> >( instr );
                    break;
                case CPHVB_MODF*100+CPHVB_UINT8:
                    traverse_3<cphvb_uint8, modf_functor<cphvb_uint8,cphvb_uint8,cphvb_uint8> >( instr );
                    break;
                case CPHVB_MODF*100+CPHVB_UINT16:
                    traverse_3<cphvb_uint16, modf_functor<cphvb_uint16,cphvb_uint16,cphvb_uint16> >( instr );
                    break;
                case CPHVB_MODF*100+CPHVB_UINT32:
                    traverse_3<cphvb_uint32, modf_functor<cphvb_uint32,cphvb_uint32,cphvb_uint32> >( instr );
                    break;
                case CPHVB_MODF*100+CPHVB_UINT64:
                    traverse_3<cphvb_uint64, modf_functor<cphvb_uint64,cphvb_uint64,cphvb_uint64> >( instr );
                    break;
                case CPHVB_MODF*100+CPHVB_FLOAT32:
                    traverse_3<cphvb_float32, modf_functor<cphvb_float32,cphvb_float32,cphvb_float32> >( instr );
                    break;
                case CPHVB_MODF*100+CPHVB_FLOAT64:
                    traverse_3<cphvb_float64, modf_functor<cphvb_float64,cphvb_float64,cphvb_float64> >( instr );
                    break;
                case CPHVB_FREXP*100+CPHVB_BOOL:
                    traverse_3<cphvb_bool, frexp_functor<cphvb_bool,cphvb_bool,cphvb_bool> >( instr );
                    break;
                case CPHVB_FREXP*100+CPHVB_INT8:
                    traverse_3<cphvb_int8, frexp_functor<cphvb_int8,cphvb_int8,cphvb_int8> >( instr );
                    break;
                case CPHVB_FREXP*100+CPHVB_INT16:
                    traverse_3<cphvb_int16, frexp_functor<cphvb_int16,cphvb_int16,cphvb_int16> >( instr );
                    break;
                case CPHVB_FREXP*100+CPHVB_INT32:
                    traverse_3<cphvb_int32, frexp_functor<cphvb_int32,cphvb_int32,cphvb_int32> >( instr );
                    break;
                case CPHVB_FREXP*100+CPHVB_INT64:
                    traverse_3<cphvb_int64, frexp_functor<cphvb_int64,cphvb_int64,cphvb_int64> >( instr );
                    break;
                case CPHVB_FREXP*100+CPHVB_UINT8:
                    traverse_3<cphvb_uint8, frexp_functor<cphvb_uint8,cphvb_uint8,cphvb_uint8> >( instr );
                    break;
                case CPHVB_FREXP*100+CPHVB_UINT16:
                    traverse_3<cphvb_uint16, frexp_functor<cphvb_uint16,cphvb_uint16,cphvb_uint16> >( instr );
                    break;
                case CPHVB_FREXP*100+CPHVB_UINT32:
                    traverse_3<cphvb_uint32, frexp_functor<cphvb_uint32,cphvb_uint32,cphvb_uint32> >( instr );
                    break;
                case CPHVB_FREXP*100+CPHVB_UINT64:
                    traverse_3<cphvb_uint64, frexp_functor<cphvb_uint64,cphvb_uint64,cphvb_uint64> >( instr );
                    break;
                case CPHVB_FREXP*100+CPHVB_FLOAT32:
                    traverse_3<cphvb_float32, frexp_functor<cphvb_float32,cphvb_float32,cphvb_float32> >( instr );
                    break;
                case CPHVB_FREXP*100+CPHVB_FLOAT64:
                    traverse_3<cphvb_float64, frexp_functor<cphvb_float64,cphvb_float64,cphvb_float64> >( instr );
                    break;
                case CPHVB_RANDOM*100+CPHVB_BOOL:
                    traverse_1<cphvb_bool, random_functor<cphvb_bool> >( instr );
                    break;
                case CPHVB_RANDOM*100+CPHVB_INT8:
                    traverse_1<cphvb_int8, random_functor<cphvb_int8> >( instr );
                    break;
                case CPHVB_RANDOM*100+CPHVB_INT16:
                    traverse_1<cphvb_int16, random_functor<cphvb_int16> >( instr );
                    break;
                case CPHVB_RANDOM*100+CPHVB_INT32:
                    traverse_1<cphvb_int32, random_functor<cphvb_int32> >( instr );
                    break;
                case CPHVB_RANDOM*100+CPHVB_INT64:
                    traverse_1<cphvb_int64, random_functor<cphvb_int64> >( instr );
                    break;
                case CPHVB_RANDOM*100+CPHVB_UINT8:
                    traverse_1<cphvb_uint8, random_functor<cphvb_uint8> >( instr );
                    break;
                case CPHVB_RANDOM*100+CPHVB_UINT16:
                    traverse_1<cphvb_uint16, random_functor<cphvb_uint16> >( instr );
                    break;
                case CPHVB_RANDOM*100+CPHVB_UINT32:
                    traverse_1<cphvb_uint32, random_functor<cphvb_uint32> >( instr );
                    break;
                case CPHVB_RANDOM*100+CPHVB_UINT64:
                    traverse_1<cphvb_uint64, random_functor<cphvb_uint64> >( instr );
                    break;
                case CPHVB_RANDOM*100+CPHVB_FLOAT32:
                    traverse_1<cphvb_float32, random_functor<cphvb_float32> >( instr );
                    break;
                case CPHVB_RANDOM*100+CPHVB_FLOAT64:
                    traverse_1<cphvb_float64, random_functor<cphvb_float64> >( instr );
                    break;
                case CPHVB_ARANGE*100+CPHVB_BOOL:
                    traverse_1<cphvb_bool, arange_functor<cphvb_bool> >( instr );
                    break;
                case CPHVB_ARANGE*100+CPHVB_INT8:
                    traverse_1<cphvb_int8, arange_functor<cphvb_int8> >( instr );
                    break;
                case CPHVB_ARANGE*100+CPHVB_INT16:
                    traverse_1<cphvb_int16, arange_functor<cphvb_int16> >( instr );
                    break;
                case CPHVB_ARANGE*100+CPHVB_INT32:
                    traverse_1<cphvb_int32, arange_functor<cphvb_int32> >( instr );
                    break;
                case CPHVB_ARANGE*100+CPHVB_INT64:
                    traverse_1<cphvb_int64, arange_functor<cphvb_int64> >( instr );
                    break;
                case CPHVB_ARANGE*100+CPHVB_UINT8:
                    traverse_1<cphvb_uint8, arange_functor<cphvb_uint8> >( instr );
                    break;
                case CPHVB_ARANGE*100+CPHVB_UINT16:
                    traverse_1<cphvb_uint16, arange_functor<cphvb_uint16> >( instr );
                    break;
                case CPHVB_ARANGE*100+CPHVB_UINT32:
                    traverse_1<cphvb_uint32, arange_functor<cphvb_uint32> >( instr );
                    break;
                case CPHVB_ARANGE*100+CPHVB_UINT64:
                    traverse_1<cphvb_uint64, arange_functor<cphvb_uint64> >( instr );
                    break;
                case CPHVB_ARANGE*100+CPHVB_FLOAT32:
                    traverse_1<cphvb_float32, arange_functor<cphvb_float32> >( instr );
                    break;
                case CPHVB_ARANGE*100+CPHVB_FLOAT64:
                    traverse_1<cphvb_float64, arange_functor<cphvb_float64> >( instr );
                    break;

                default:                // Unsupported instruction
                    instr->status = CPHVB_TYPE_NOT_SUPPORTED;
                    return CPHVB_PARTIAL_SUCCESS;

            }

    }

    return res;

}

