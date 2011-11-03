#include "stdio.h" 

#include <cphvb.h>
#include "score_dispatch.h"
#include "score_iter.hpp"
#include "score_funcs.hpp"

cphvb_error dispatch( cphvb_instruction *instr ) {

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
                    iter<cphvb_bool>( instr, &score_add );
                    break;
                case CPHVB_ADD*100+CPHVB_INT8:
                    iter<cphvb_int8>( instr, &score_add );
                    break;
                case CPHVB_ADD*100+CPHVB_INT16:
                    iter<cphvb_int16>( instr, &score_add );
                    break;
                case CPHVB_ADD*100+CPHVB_INT32:
                    iter<cphvb_int32>( instr, &score_add );
                    break;
                case CPHVB_ADD*100+CPHVB_INT64:
                    iter<cphvb_int64>( instr, &score_add );
                    break;
                case CPHVB_ADD*100+CPHVB_UINT8:
                    iter<cphvb_uint8>( instr, &score_add );
                    break;
                case CPHVB_ADD*100+CPHVB_UINT16:
                    iter<cphvb_uint16>( instr, &score_add );
                    break;
                case CPHVB_ADD*100+CPHVB_UINT32:
                    iter<cphvb_uint32>( instr, &score_add );
                    break;
                case CPHVB_ADD*100+CPHVB_UINT64:
                    iter<cphvb_uint64>( instr, &score_add );
                    break;
                case CPHVB_ADD*100+CPHVB_FLOAT32:
                    iter<cphvb_float32>( instr, &score_add );
                    break;
                case CPHVB_ADD*100+CPHVB_FLOAT64:
                    iter<cphvb_float64>( instr, &score_add );
                    break;
                case CPHVB_SUBTRACT*100+CPHVB_BOOL:
                    iter<cphvb_bool>( instr, &score_subtract );
                    break;
                case CPHVB_SUBTRACT*100+CPHVB_INT8:
                    iter<cphvb_int8>( instr, &score_subtract );
                    break;
                case CPHVB_SUBTRACT*100+CPHVB_INT16:
                    iter<cphvb_int16>( instr, &score_subtract );
                    break;
                case CPHVB_SUBTRACT*100+CPHVB_INT32:
                    iter<cphvb_int32>( instr, &score_subtract );
                    break;
                case CPHVB_SUBTRACT*100+CPHVB_INT64:
                    iter<cphvb_int64>( instr, &score_subtract );
                    break;
                case CPHVB_SUBTRACT*100+CPHVB_UINT8:
                    iter<cphvb_uint8>( instr, &score_subtract );
                    break;
                case CPHVB_SUBTRACT*100+CPHVB_UINT16:
                    iter<cphvb_uint16>( instr, &score_subtract );
                    break;
                case CPHVB_SUBTRACT*100+CPHVB_UINT32:
                    iter<cphvb_uint32>( instr, &score_subtract );
                    break;
                case CPHVB_SUBTRACT*100+CPHVB_UINT64:
                    iter<cphvb_uint64>( instr, &score_subtract );
                    break;
                case CPHVB_SUBTRACT*100+CPHVB_FLOAT32:
                    iter<cphvb_float32>( instr, &score_subtract );
                    break;
                case CPHVB_SUBTRACT*100+CPHVB_FLOAT64:
                    iter<cphvb_float64>( instr, &score_subtract );
                    break;
                case CPHVB_MULTIPLY*100+CPHVB_BOOL:
                    iter<cphvb_bool>( instr, &score_multiply );
                    break;
                case CPHVB_MULTIPLY*100+CPHVB_INT8:
                    iter<cphvb_int8>( instr, &score_multiply );
                    break;
                case CPHVB_MULTIPLY*100+CPHVB_INT16:
                    iter<cphvb_int16>( instr, &score_multiply );
                    break;
                case CPHVB_MULTIPLY*100+CPHVB_INT32:
                    iter<cphvb_int32>( instr, &score_multiply );
                    break;
                case CPHVB_MULTIPLY*100+CPHVB_INT64:
                    iter<cphvb_int64>( instr, &score_multiply );
                    break;
                case CPHVB_MULTIPLY*100+CPHVB_UINT8:
                    iter<cphvb_uint8>( instr, &score_multiply );
                    break;
                case CPHVB_MULTIPLY*100+CPHVB_UINT16:
                    iter<cphvb_uint16>( instr, &score_multiply );
                    break;
                case CPHVB_MULTIPLY*100+CPHVB_UINT32:
                    iter<cphvb_uint32>( instr, &score_multiply );
                    break;
                case CPHVB_MULTIPLY*100+CPHVB_UINT64:
                    iter<cphvb_uint64>( instr, &score_multiply );
                    break;
                case CPHVB_MULTIPLY*100+CPHVB_FLOAT32:
                    iter<cphvb_float32>( instr, &score_multiply );
                    break;
                case CPHVB_MULTIPLY*100+CPHVB_FLOAT64:
                    iter<cphvb_float64>( instr, &score_multiply );
                    break;
                case CPHVB_DIVIDE*100+CPHVB_BOOL:
                    iter<cphvb_bool>( instr, &score_divide );
                    break;
                case CPHVB_DIVIDE*100+CPHVB_INT8:
                    iter<cphvb_int8>( instr, &score_divide );
                    break;
                case CPHVB_DIVIDE*100+CPHVB_INT16:
                    iter<cphvb_int16>( instr, &score_divide );
                    break;
                case CPHVB_DIVIDE*100+CPHVB_INT32:
                    iter<cphvb_int32>( instr, &score_divide );
                    break;
                case CPHVB_DIVIDE*100+CPHVB_INT64:
                    iter<cphvb_int64>( instr, &score_divide );
                    break;
                case CPHVB_DIVIDE*100+CPHVB_UINT8:
                    iter<cphvb_uint8>( instr, &score_divide );
                    break;
                case CPHVB_DIVIDE*100+CPHVB_UINT16:
                    iter<cphvb_uint16>( instr, &score_divide );
                    break;
                case CPHVB_DIVIDE*100+CPHVB_UINT32:
                    iter<cphvb_uint32>( instr, &score_divide );
                    break;
                case CPHVB_DIVIDE*100+CPHVB_UINT64:
                    iter<cphvb_uint64>( instr, &score_divide );
                    break;
                case CPHVB_DIVIDE*100+CPHVB_FLOAT32:
                    iter<cphvb_float32>( instr, &score_divide );
                    break;
                case CPHVB_DIVIDE*100+CPHVB_FLOAT64:
                    iter<cphvb_float64>( instr, &score_divide );
                    break;
                case CPHVB_LOGADDEXP*100+CPHVB_BOOL:
                    iter<cphvb_bool>( instr, &score_logaddexp );
                    break;
                case CPHVB_LOGADDEXP*100+CPHVB_INT8:
                    iter<cphvb_int8>( instr, &score_logaddexp );
                    break;
                case CPHVB_LOGADDEXP*100+CPHVB_INT16:
                    iter<cphvb_int16>( instr, &score_logaddexp );
                    break;
                case CPHVB_LOGADDEXP*100+CPHVB_INT32:
                    iter<cphvb_int32>( instr, &score_logaddexp );
                    break;
                case CPHVB_LOGADDEXP*100+CPHVB_INT64:
                    iter<cphvb_int64>( instr, &score_logaddexp );
                    break;
                case CPHVB_LOGADDEXP*100+CPHVB_UINT8:
                    iter<cphvb_uint8>( instr, &score_logaddexp );
                    break;
                case CPHVB_LOGADDEXP*100+CPHVB_UINT16:
                    iter<cphvb_uint16>( instr, &score_logaddexp );
                    break;
                case CPHVB_LOGADDEXP*100+CPHVB_UINT32:
                    iter<cphvb_uint32>( instr, &score_logaddexp );
                    break;
                case CPHVB_LOGADDEXP*100+CPHVB_UINT64:
                    iter<cphvb_uint64>( instr, &score_logaddexp );
                    break;
                case CPHVB_LOGADDEXP*100+CPHVB_FLOAT32:
                    iter<cphvb_float32>( instr, &score_logaddexp );
                    break;
                case CPHVB_LOGADDEXP*100+CPHVB_FLOAT64:
                    iter<cphvb_float64>( instr, &score_logaddexp );
                    break;
                case CPHVB_LOGADDEXP2*100+CPHVB_BOOL:
                    iter<cphvb_bool>( instr, &score_logaddexp2 );
                    break;
                case CPHVB_LOGADDEXP2*100+CPHVB_INT8:
                    iter<cphvb_int8>( instr, &score_logaddexp2 );
                    break;
                case CPHVB_LOGADDEXP2*100+CPHVB_INT16:
                    iter<cphvb_int16>( instr, &score_logaddexp2 );
                    break;
                case CPHVB_LOGADDEXP2*100+CPHVB_INT32:
                    iter<cphvb_int32>( instr, &score_logaddexp2 );
                    break;
                case CPHVB_LOGADDEXP2*100+CPHVB_INT64:
                    iter<cphvb_int64>( instr, &score_logaddexp2 );
                    break;
                case CPHVB_LOGADDEXP2*100+CPHVB_UINT8:
                    iter<cphvb_uint8>( instr, &score_logaddexp2 );
                    break;
                case CPHVB_LOGADDEXP2*100+CPHVB_UINT16:
                    iter<cphvb_uint16>( instr, &score_logaddexp2 );
                    break;
                case CPHVB_LOGADDEXP2*100+CPHVB_UINT32:
                    iter<cphvb_uint32>( instr, &score_logaddexp2 );
                    break;
                case CPHVB_LOGADDEXP2*100+CPHVB_UINT64:
                    iter<cphvb_uint64>( instr, &score_logaddexp2 );
                    break;
                case CPHVB_LOGADDEXP2*100+CPHVB_FLOAT32:
                    iter<cphvb_float32>( instr, &score_logaddexp2 );
                    break;
                case CPHVB_LOGADDEXP2*100+CPHVB_FLOAT64:
                    iter<cphvb_float64>( instr, &score_logaddexp2 );
                    break;
                case CPHVB_TRUE_DIVIDE*100+CPHVB_BOOL:
                    iter<cphvb_bool>( instr, &score_true_divide );
                    break;
                case CPHVB_TRUE_DIVIDE*100+CPHVB_INT8:
                    iter<cphvb_int8>( instr, &score_true_divide );
                    break;
                case CPHVB_TRUE_DIVIDE*100+CPHVB_INT16:
                    iter<cphvb_int16>( instr, &score_true_divide );
                    break;
                case CPHVB_TRUE_DIVIDE*100+CPHVB_INT32:
                    iter<cphvb_int32>( instr, &score_true_divide );
                    break;
                case CPHVB_TRUE_DIVIDE*100+CPHVB_INT64:
                    iter<cphvb_int64>( instr, &score_true_divide );
                    break;
                case CPHVB_TRUE_DIVIDE*100+CPHVB_UINT8:
                    iter<cphvb_uint8>( instr, &score_true_divide );
                    break;
                case CPHVB_TRUE_DIVIDE*100+CPHVB_UINT16:
                    iter<cphvb_uint16>( instr, &score_true_divide );
                    break;
                case CPHVB_TRUE_DIVIDE*100+CPHVB_UINT32:
                    iter<cphvb_uint32>( instr, &score_true_divide );
                    break;
                case CPHVB_TRUE_DIVIDE*100+CPHVB_UINT64:
                    iter<cphvb_uint64>( instr, &score_true_divide );
                    break;
                case CPHVB_TRUE_DIVIDE*100+CPHVB_FLOAT32:
                    iter<cphvb_float32>( instr, &score_true_divide );
                    break;
                case CPHVB_TRUE_DIVIDE*100+CPHVB_FLOAT64:
                    iter<cphvb_float64>( instr, &score_true_divide );
                    break;
                case CPHVB_FLOOR_DIVIDE*100+CPHVB_BOOL:
                    iter<cphvb_bool>( instr, &score_floor_divide );
                    break;
                case CPHVB_FLOOR_DIVIDE*100+CPHVB_INT8:
                    iter<cphvb_int8>( instr, &score_floor_divide );
                    break;
                case CPHVB_FLOOR_DIVIDE*100+CPHVB_INT16:
                    iter<cphvb_int16>( instr, &score_floor_divide );
                    break;
                case CPHVB_FLOOR_DIVIDE*100+CPHVB_INT32:
                    iter<cphvb_int32>( instr, &score_floor_divide );
                    break;
                case CPHVB_FLOOR_DIVIDE*100+CPHVB_INT64:
                    iter<cphvb_int64>( instr, &score_floor_divide );
                    break;
                case CPHVB_FLOOR_DIVIDE*100+CPHVB_UINT8:
                    iter<cphvb_uint8>( instr, &score_floor_divide );
                    break;
                case CPHVB_FLOOR_DIVIDE*100+CPHVB_UINT16:
                    iter<cphvb_uint16>( instr, &score_floor_divide );
                    break;
                case CPHVB_FLOOR_DIVIDE*100+CPHVB_UINT32:
                    iter<cphvb_uint32>( instr, &score_floor_divide );
                    break;
                case CPHVB_FLOOR_DIVIDE*100+CPHVB_UINT64:
                    iter<cphvb_uint64>( instr, &score_floor_divide );
                    break;
                case CPHVB_FLOOR_DIVIDE*100+CPHVB_FLOAT32:
                    iter<cphvb_float32>( instr, &score_floor_divide );
                    break;
                case CPHVB_FLOOR_DIVIDE*100+CPHVB_FLOAT64:
                    iter<cphvb_float64>( instr, &score_floor_divide );
                    break;
                case CPHVB_POWER*100+CPHVB_BOOL:
                    iter<cphvb_bool>( instr, &score_power );
                    break;
                case CPHVB_POWER*100+CPHVB_INT8:
                    iter<cphvb_int8>( instr, &score_power );
                    break;
                case CPHVB_POWER*100+CPHVB_INT16:
                    iter<cphvb_int16>( instr, &score_power );
                    break;
                case CPHVB_POWER*100+CPHVB_INT32:
                    iter<cphvb_int32>( instr, &score_power );
                    break;
                case CPHVB_POWER*100+CPHVB_INT64:
                    iter<cphvb_int64>( instr, &score_power );
                    break;
                case CPHVB_POWER*100+CPHVB_UINT8:
                    iter<cphvb_uint8>( instr, &score_power );
                    break;
                case CPHVB_POWER*100+CPHVB_UINT16:
                    iter<cphvb_uint16>( instr, &score_power );
                    break;
                case CPHVB_POWER*100+CPHVB_UINT32:
                    iter<cphvb_uint32>( instr, &score_power );
                    break;
                case CPHVB_POWER*100+CPHVB_UINT64:
                    iter<cphvb_uint64>( instr, &score_power );
                    break;
                case CPHVB_POWER*100+CPHVB_FLOAT32:
                    iter<cphvb_float32>( instr, &score_power );
                    break;
                case CPHVB_POWER*100+CPHVB_FLOAT64:
                    iter<cphvb_float64>( instr, &score_power );
                    break;
                case CPHVB_REMAINDER*100+CPHVB_BOOL:
                    iter<cphvb_bool>( instr, &score_remainder );
                    break;
                case CPHVB_REMAINDER*100+CPHVB_INT8:
                    iter<cphvb_int8>( instr, &score_remainder );
                    break;
                case CPHVB_REMAINDER*100+CPHVB_INT16:
                    iter<cphvb_int16>( instr, &score_remainder );
                    break;
                case CPHVB_REMAINDER*100+CPHVB_INT32:
                    iter<cphvb_int32>( instr, &score_remainder );
                    break;
                case CPHVB_REMAINDER*100+CPHVB_INT64:
                    iter<cphvb_int64>( instr, &score_remainder );
                    break;
                case CPHVB_REMAINDER*100+CPHVB_UINT8:
                    iter<cphvb_uint8>( instr, &score_remainder );
                    break;
                case CPHVB_REMAINDER*100+CPHVB_UINT16:
                    iter<cphvb_uint16>( instr, &score_remainder );
                    break;
                case CPHVB_REMAINDER*100+CPHVB_UINT32:
                    iter<cphvb_uint32>( instr, &score_remainder );
                    break;
                case CPHVB_REMAINDER*100+CPHVB_UINT64:
                    iter<cphvb_uint64>( instr, &score_remainder );
                    break;
                case CPHVB_REMAINDER*100+CPHVB_FLOAT32:
                    iter<cphvb_float32>( instr, &score_remainder );
                    break;
                case CPHVB_REMAINDER*100+CPHVB_FLOAT64:
                    iter<cphvb_float64>( instr, &score_remainder );
                    break;
                case CPHVB_MOD*100+CPHVB_BOOL:
                    iter<cphvb_bool>( instr, &score_mod );
                    break;
                case CPHVB_MOD*100+CPHVB_INT8:
                    iter<cphvb_int8>( instr, &score_mod );
                    break;
                case CPHVB_MOD*100+CPHVB_INT16:
                    iter<cphvb_int16>( instr, &score_mod );
                    break;
                case CPHVB_MOD*100+CPHVB_INT32:
                    iter<cphvb_int32>( instr, &score_mod );
                    break;
                case CPHVB_MOD*100+CPHVB_INT64:
                    iter<cphvb_int64>( instr, &score_mod );
                    break;
                case CPHVB_MOD*100+CPHVB_UINT8:
                    iter<cphvb_uint8>( instr, &score_mod );
                    break;
                case CPHVB_MOD*100+CPHVB_UINT16:
                    iter<cphvb_uint16>( instr, &score_mod );
                    break;
                case CPHVB_MOD*100+CPHVB_UINT32:
                    iter<cphvb_uint32>( instr, &score_mod );
                    break;
                case CPHVB_MOD*100+CPHVB_UINT64:
                    iter<cphvb_uint64>( instr, &score_mod );
                    break;
                case CPHVB_MOD*100+CPHVB_FLOAT32:
                    iter<cphvb_float32>( instr, &score_mod );
                    break;
                case CPHVB_MOD*100+CPHVB_FLOAT64:
                    iter<cphvb_float64>( instr, &score_mod );
                    break;
                case CPHVB_FMOD*100+CPHVB_BOOL:
                    iter<cphvb_bool>( instr, &score_fmod );
                    break;
                case CPHVB_FMOD*100+CPHVB_INT8:
                    iter<cphvb_int8>( instr, &score_fmod );
                    break;
                case CPHVB_FMOD*100+CPHVB_INT16:
                    iter<cphvb_int16>( instr, &score_fmod );
                    break;
                case CPHVB_FMOD*100+CPHVB_INT32:
                    iter<cphvb_int32>( instr, &score_fmod );
                    break;
                case CPHVB_FMOD*100+CPHVB_INT64:
                    iter<cphvb_int64>( instr, &score_fmod );
                    break;
                case CPHVB_FMOD*100+CPHVB_UINT8:
                    iter<cphvb_uint8>( instr, &score_fmod );
                    break;
                case CPHVB_FMOD*100+CPHVB_UINT16:
                    iter<cphvb_uint16>( instr, &score_fmod );
                    break;
                case CPHVB_FMOD*100+CPHVB_UINT32:
                    iter<cphvb_uint32>( instr, &score_fmod );
                    break;
                case CPHVB_FMOD*100+CPHVB_UINT64:
                    iter<cphvb_uint64>( instr, &score_fmod );
                    break;
                case CPHVB_FMOD*100+CPHVB_FLOAT32:
                    iter<cphvb_float32>( instr, &score_fmod );
                    break;
                case CPHVB_FMOD*100+CPHVB_FLOAT64:
                    iter<cphvb_float64>( instr, &score_fmod );
                    break;
                case CPHVB_BITWISE_AND*100+CPHVB_BOOL:
                    iter<cphvb_bool>( instr, &score_bitwise_and );
                    break;
                case CPHVB_BITWISE_AND*100+CPHVB_INT8:
                    iter<cphvb_int8>( instr, &score_bitwise_and );
                    break;
                case CPHVB_BITWISE_AND*100+CPHVB_INT16:
                    iter<cphvb_int16>( instr, &score_bitwise_and );
                    break;
                case CPHVB_BITWISE_AND*100+CPHVB_INT32:
                    iter<cphvb_int32>( instr, &score_bitwise_and );
                    break;
                case CPHVB_BITWISE_AND*100+CPHVB_INT64:
                    iter<cphvb_int64>( instr, &score_bitwise_and );
                    break;
                case CPHVB_BITWISE_AND*100+CPHVB_UINT8:
                    iter<cphvb_uint8>( instr, &score_bitwise_and );
                    break;
                case CPHVB_BITWISE_AND*100+CPHVB_UINT16:
                    iter<cphvb_uint16>( instr, &score_bitwise_and );
                    break;
                case CPHVB_BITWISE_AND*100+CPHVB_UINT32:
                    iter<cphvb_uint32>( instr, &score_bitwise_and );
                    break;
                case CPHVB_BITWISE_AND*100+CPHVB_UINT64:
                    iter<cphvb_uint64>( instr, &score_bitwise_and );
                    break;
                case CPHVB_BITWISE_AND*100+CPHVB_FLOAT32:
                    iter<cphvb_float32>( instr, &score_bitwise_and );
                    break;
                case CPHVB_BITWISE_AND*100+CPHVB_FLOAT64:
                    iter<cphvb_float64>( instr, &score_bitwise_and );
                    break;
                case CPHVB_BITWISE_OR*100+CPHVB_BOOL:
                    iter<cphvb_bool>( instr, &score_bitwise_or );
                    break;
                case CPHVB_BITWISE_OR*100+CPHVB_INT8:
                    iter<cphvb_int8>( instr, &score_bitwise_or );
                    break;
                case CPHVB_BITWISE_OR*100+CPHVB_INT16:
                    iter<cphvb_int16>( instr, &score_bitwise_or );
                    break;
                case CPHVB_BITWISE_OR*100+CPHVB_INT32:
                    iter<cphvb_int32>( instr, &score_bitwise_or );
                    break;
                case CPHVB_BITWISE_OR*100+CPHVB_INT64:
                    iter<cphvb_int64>( instr, &score_bitwise_or );
                    break;
                case CPHVB_BITWISE_OR*100+CPHVB_UINT8:
                    iter<cphvb_uint8>( instr, &score_bitwise_or );
                    break;
                case CPHVB_BITWISE_OR*100+CPHVB_UINT16:
                    iter<cphvb_uint16>( instr, &score_bitwise_or );
                    break;
                case CPHVB_BITWISE_OR*100+CPHVB_UINT32:
                    iter<cphvb_uint32>( instr, &score_bitwise_or );
                    break;
                case CPHVB_BITWISE_OR*100+CPHVB_UINT64:
                    iter<cphvb_uint64>( instr, &score_bitwise_or );
                    break;
                case CPHVB_BITWISE_OR*100+CPHVB_FLOAT32:
                    iter<cphvb_float32>( instr, &score_bitwise_or );
                    break;
                case CPHVB_BITWISE_OR*100+CPHVB_FLOAT64:
                    iter<cphvb_float64>( instr, &score_bitwise_or );
                    break;
                case CPHVB_BITWISE_XOR*100+CPHVB_BOOL:
                    iter<cphvb_bool>( instr, &score_bitwise_xor );
                    break;
                case CPHVB_BITWISE_XOR*100+CPHVB_INT8:
                    iter<cphvb_int8>( instr, &score_bitwise_xor );
                    break;
                case CPHVB_BITWISE_XOR*100+CPHVB_INT16:
                    iter<cphvb_int16>( instr, &score_bitwise_xor );
                    break;
                case CPHVB_BITWISE_XOR*100+CPHVB_INT32:
                    iter<cphvb_int32>( instr, &score_bitwise_xor );
                    break;
                case CPHVB_BITWISE_XOR*100+CPHVB_INT64:
                    iter<cphvb_int64>( instr, &score_bitwise_xor );
                    break;
                case CPHVB_BITWISE_XOR*100+CPHVB_UINT8:
                    iter<cphvb_uint8>( instr, &score_bitwise_xor );
                    break;
                case CPHVB_BITWISE_XOR*100+CPHVB_UINT16:
                    iter<cphvb_uint16>( instr, &score_bitwise_xor );
                    break;
                case CPHVB_BITWISE_XOR*100+CPHVB_UINT32:
                    iter<cphvb_uint32>( instr, &score_bitwise_xor );
                    break;
                case CPHVB_BITWISE_XOR*100+CPHVB_UINT64:
                    iter<cphvb_uint64>( instr, &score_bitwise_xor );
                    break;
                case CPHVB_BITWISE_XOR*100+CPHVB_FLOAT32:
                    iter<cphvb_float32>( instr, &score_bitwise_xor );
                    break;
                case CPHVB_BITWISE_XOR*100+CPHVB_FLOAT64:
                    iter<cphvb_float64>( instr, &score_bitwise_xor );
                    break;
                case CPHVB_LOGICAL_AND*100+CPHVB_BOOL:
                    iter<cphvb_bool>( instr, &score_logical_and );
                    break;
                case CPHVB_LOGICAL_AND*100+CPHVB_INT8:
                    iter<cphvb_int8>( instr, &score_logical_and );
                    break;
                case CPHVB_LOGICAL_AND*100+CPHVB_INT16:
                    iter<cphvb_int16>( instr, &score_logical_and );
                    break;
                case CPHVB_LOGICAL_AND*100+CPHVB_INT32:
                    iter<cphvb_int32>( instr, &score_logical_and );
                    break;
                case CPHVB_LOGICAL_AND*100+CPHVB_INT64:
                    iter<cphvb_int64>( instr, &score_logical_and );
                    break;
                case CPHVB_LOGICAL_AND*100+CPHVB_UINT8:
                    iter<cphvb_uint8>( instr, &score_logical_and );
                    break;
                case CPHVB_LOGICAL_AND*100+CPHVB_UINT16:
                    iter<cphvb_uint16>( instr, &score_logical_and );
                    break;
                case CPHVB_LOGICAL_AND*100+CPHVB_UINT32:
                    iter<cphvb_uint32>( instr, &score_logical_and );
                    break;
                case CPHVB_LOGICAL_AND*100+CPHVB_UINT64:
                    iter<cphvb_uint64>( instr, &score_logical_and );
                    break;
                case CPHVB_LOGICAL_AND*100+CPHVB_FLOAT32:
                    iter<cphvb_float32>( instr, &score_logical_and );
                    break;
                case CPHVB_LOGICAL_AND*100+CPHVB_FLOAT64:
                    iter<cphvb_float64>( instr, &score_logical_and );
                    break;
                case CPHVB_LOGICAL_OR*100+CPHVB_BOOL:
                    iter<cphvb_bool>( instr, &score_logical_or );
                    break;
                case CPHVB_LOGICAL_OR*100+CPHVB_INT8:
                    iter<cphvb_int8>( instr, &score_logical_or );
                    break;
                case CPHVB_LOGICAL_OR*100+CPHVB_INT16:
                    iter<cphvb_int16>( instr, &score_logical_or );
                    break;
                case CPHVB_LOGICAL_OR*100+CPHVB_INT32:
                    iter<cphvb_int32>( instr, &score_logical_or );
                    break;
                case CPHVB_LOGICAL_OR*100+CPHVB_INT64:
                    iter<cphvb_int64>( instr, &score_logical_or );
                    break;
                case CPHVB_LOGICAL_OR*100+CPHVB_UINT8:
                    iter<cphvb_uint8>( instr, &score_logical_or );
                    break;
                case CPHVB_LOGICAL_OR*100+CPHVB_UINT16:
                    iter<cphvb_uint16>( instr, &score_logical_or );
                    break;
                case CPHVB_LOGICAL_OR*100+CPHVB_UINT32:
                    iter<cphvb_uint32>( instr, &score_logical_or );
                    break;
                case CPHVB_LOGICAL_OR*100+CPHVB_UINT64:
                    iter<cphvb_uint64>( instr, &score_logical_or );
                    break;
                case CPHVB_LOGICAL_OR*100+CPHVB_FLOAT32:
                    iter<cphvb_float32>( instr, &score_logical_or );
                    break;
                case CPHVB_LOGICAL_OR*100+CPHVB_FLOAT64:
                    iter<cphvb_float64>( instr, &score_logical_or );
                    break;
                case CPHVB_LOGICAL_XOR*100+CPHVB_BOOL:
                    iter<cphvb_bool>( instr, &score_logical_xor );
                    break;
                case CPHVB_LOGICAL_XOR*100+CPHVB_INT8:
                    iter<cphvb_int8>( instr, &score_logical_xor );
                    break;
                case CPHVB_LOGICAL_XOR*100+CPHVB_INT16:
                    iter<cphvb_int16>( instr, &score_logical_xor );
                    break;
                case CPHVB_LOGICAL_XOR*100+CPHVB_INT32:
                    iter<cphvb_int32>( instr, &score_logical_xor );
                    break;
                case CPHVB_LOGICAL_XOR*100+CPHVB_INT64:
                    iter<cphvb_int64>( instr, &score_logical_xor );
                    break;
                case CPHVB_LOGICAL_XOR*100+CPHVB_UINT8:
                    iter<cphvb_uint8>( instr, &score_logical_xor );
                    break;
                case CPHVB_LOGICAL_XOR*100+CPHVB_UINT16:
                    iter<cphvb_uint16>( instr, &score_logical_xor );
                    break;
                case CPHVB_LOGICAL_XOR*100+CPHVB_UINT32:
                    iter<cphvb_uint32>( instr, &score_logical_xor );
                    break;
                case CPHVB_LOGICAL_XOR*100+CPHVB_UINT64:
                    iter<cphvb_uint64>( instr, &score_logical_xor );
                    break;
                case CPHVB_LOGICAL_XOR*100+CPHVB_FLOAT32:
                    iter<cphvb_float32>( instr, &score_logical_xor );
                    break;
                case CPHVB_LOGICAL_XOR*100+CPHVB_FLOAT64:
                    iter<cphvb_float64>( instr, &score_logical_xor );
                    break;
                case CPHVB_LEFT_SHIFT*100+CPHVB_BOOL:
                    iter<cphvb_bool>( instr, &score_left_shift );
                    break;
                case CPHVB_LEFT_SHIFT*100+CPHVB_INT8:
                    iter<cphvb_int8>( instr, &score_left_shift );
                    break;
                case CPHVB_LEFT_SHIFT*100+CPHVB_INT16:
                    iter<cphvb_int16>( instr, &score_left_shift );
                    break;
                case CPHVB_LEFT_SHIFT*100+CPHVB_INT32:
                    iter<cphvb_int32>( instr, &score_left_shift );
                    break;
                case CPHVB_LEFT_SHIFT*100+CPHVB_INT64:
                    iter<cphvb_int64>( instr, &score_left_shift );
                    break;
                case CPHVB_LEFT_SHIFT*100+CPHVB_UINT8:
                    iter<cphvb_uint8>( instr, &score_left_shift );
                    break;
                case CPHVB_LEFT_SHIFT*100+CPHVB_UINT16:
                    iter<cphvb_uint16>( instr, &score_left_shift );
                    break;
                case CPHVB_LEFT_SHIFT*100+CPHVB_UINT32:
                    iter<cphvb_uint32>( instr, &score_left_shift );
                    break;
                case CPHVB_LEFT_SHIFT*100+CPHVB_UINT64:
                    iter<cphvb_uint64>( instr, &score_left_shift );
                    break;
                case CPHVB_LEFT_SHIFT*100+CPHVB_FLOAT32:
                    iter<cphvb_float32>( instr, &score_left_shift );
                    break;
                case CPHVB_LEFT_SHIFT*100+CPHVB_FLOAT64:
                    iter<cphvb_float64>( instr, &score_left_shift );
                    break;
                case CPHVB_RIGHT_SHIFT*100+CPHVB_BOOL:
                    iter<cphvb_bool>( instr, &score_right_shift );
                    break;
                case CPHVB_RIGHT_SHIFT*100+CPHVB_INT8:
                    iter<cphvb_int8>( instr, &score_right_shift );
                    break;
                case CPHVB_RIGHT_SHIFT*100+CPHVB_INT16:
                    iter<cphvb_int16>( instr, &score_right_shift );
                    break;
                case CPHVB_RIGHT_SHIFT*100+CPHVB_INT32:
                    iter<cphvb_int32>( instr, &score_right_shift );
                    break;
                case CPHVB_RIGHT_SHIFT*100+CPHVB_INT64:
                    iter<cphvb_int64>( instr, &score_right_shift );
                    break;
                case CPHVB_RIGHT_SHIFT*100+CPHVB_UINT8:
                    iter<cphvb_uint8>( instr, &score_right_shift );
                    break;
                case CPHVB_RIGHT_SHIFT*100+CPHVB_UINT16:
                    iter<cphvb_uint16>( instr, &score_right_shift );
                    break;
                case CPHVB_RIGHT_SHIFT*100+CPHVB_UINT32:
                    iter<cphvb_uint32>( instr, &score_right_shift );
                    break;
                case CPHVB_RIGHT_SHIFT*100+CPHVB_UINT64:
                    iter<cphvb_uint64>( instr, &score_right_shift );
                    break;
                case CPHVB_RIGHT_SHIFT*100+CPHVB_FLOAT32:
                    iter<cphvb_float32>( instr, &score_right_shift );
                    break;
                case CPHVB_RIGHT_SHIFT*100+CPHVB_FLOAT64:
                    iter<cphvb_float64>( instr, &score_right_shift );
                    break;
                case CPHVB_GREATER*100+CPHVB_BOOL:
                    iter<cphvb_bool>( instr, &score_greater );
                    break;
                case CPHVB_GREATER*100+CPHVB_INT8:
                    iter<cphvb_int8>( instr, &score_greater );
                    break;
                case CPHVB_GREATER*100+CPHVB_INT16:
                    iter<cphvb_int16>( instr, &score_greater );
                    break;
                case CPHVB_GREATER*100+CPHVB_INT32:
                    iter<cphvb_int32>( instr, &score_greater );
                    break;
                case CPHVB_GREATER*100+CPHVB_INT64:
                    iter<cphvb_int64>( instr, &score_greater );
                    break;
                case CPHVB_GREATER*100+CPHVB_UINT8:
                    iter<cphvb_uint8>( instr, &score_greater );
                    break;
                case CPHVB_GREATER*100+CPHVB_UINT16:
                    iter<cphvb_uint16>( instr, &score_greater );
                    break;
                case CPHVB_GREATER*100+CPHVB_UINT32:
                    iter<cphvb_uint32>( instr, &score_greater );
                    break;
                case CPHVB_GREATER*100+CPHVB_UINT64:
                    iter<cphvb_uint64>( instr, &score_greater );
                    break;
                case CPHVB_GREATER*100+CPHVB_FLOAT32:
                    iter<cphvb_float32>( instr, &score_greater );
                    break;
                case CPHVB_GREATER*100+CPHVB_FLOAT64:
                    iter<cphvb_float64>( instr, &score_greater );
                    break;
                case CPHVB_GREATER_EQUAL*100+CPHVB_BOOL:
                    iter<cphvb_bool>( instr, &score_greater_equal );
                    break;
                case CPHVB_GREATER_EQUAL*100+CPHVB_INT8:
                    iter<cphvb_int8>( instr, &score_greater_equal );
                    break;
                case CPHVB_GREATER_EQUAL*100+CPHVB_INT16:
                    iter<cphvb_int16>( instr, &score_greater_equal );
                    break;
                case CPHVB_GREATER_EQUAL*100+CPHVB_INT32:
                    iter<cphvb_int32>( instr, &score_greater_equal );
                    break;
                case CPHVB_GREATER_EQUAL*100+CPHVB_INT64:
                    iter<cphvb_int64>( instr, &score_greater_equal );
                    break;
                case CPHVB_GREATER_EQUAL*100+CPHVB_UINT8:
                    iter<cphvb_uint8>( instr, &score_greater_equal );
                    break;
                case CPHVB_GREATER_EQUAL*100+CPHVB_UINT16:
                    iter<cphvb_uint16>( instr, &score_greater_equal );
                    break;
                case CPHVB_GREATER_EQUAL*100+CPHVB_UINT32:
                    iter<cphvb_uint32>( instr, &score_greater_equal );
                    break;
                case CPHVB_GREATER_EQUAL*100+CPHVB_UINT64:
                    iter<cphvb_uint64>( instr, &score_greater_equal );
                    break;
                case CPHVB_GREATER_EQUAL*100+CPHVB_FLOAT32:
                    iter<cphvb_float32>( instr, &score_greater_equal );
                    break;
                case CPHVB_GREATER_EQUAL*100+CPHVB_FLOAT64:
                    iter<cphvb_float64>( instr, &score_greater_equal );
                    break;
                case CPHVB_LESS*100+CPHVB_BOOL:
                    iter<cphvb_bool>( instr, &score_less );
                    break;
                case CPHVB_LESS*100+CPHVB_INT8:
                    iter<cphvb_int8>( instr, &score_less );
                    break;
                case CPHVB_LESS*100+CPHVB_INT16:
                    iter<cphvb_int16>( instr, &score_less );
                    break;
                case CPHVB_LESS*100+CPHVB_INT32:
                    iter<cphvb_int32>( instr, &score_less );
                    break;
                case CPHVB_LESS*100+CPHVB_INT64:
                    iter<cphvb_int64>( instr, &score_less );
                    break;
                case CPHVB_LESS*100+CPHVB_UINT8:
                    iter<cphvb_uint8>( instr, &score_less );
                    break;
                case CPHVB_LESS*100+CPHVB_UINT16:
                    iter<cphvb_uint16>( instr, &score_less );
                    break;
                case CPHVB_LESS*100+CPHVB_UINT32:
                    iter<cphvb_uint32>( instr, &score_less );
                    break;
                case CPHVB_LESS*100+CPHVB_UINT64:
                    iter<cphvb_uint64>( instr, &score_less );
                    break;
                case CPHVB_LESS*100+CPHVB_FLOAT32:
                    iter<cphvb_float32>( instr, &score_less );
                    break;
                case CPHVB_LESS*100+CPHVB_FLOAT64:
                    iter<cphvb_float64>( instr, &score_less );
                    break;
                case CPHVB_LESS_EQUAL*100+CPHVB_BOOL:
                    iter<cphvb_bool>( instr, &score_less_equal );
                    break;
                case CPHVB_LESS_EQUAL*100+CPHVB_INT8:
                    iter<cphvb_int8>( instr, &score_less_equal );
                    break;
                case CPHVB_LESS_EQUAL*100+CPHVB_INT16:
                    iter<cphvb_int16>( instr, &score_less_equal );
                    break;
                case CPHVB_LESS_EQUAL*100+CPHVB_INT32:
                    iter<cphvb_int32>( instr, &score_less_equal );
                    break;
                case CPHVB_LESS_EQUAL*100+CPHVB_INT64:
                    iter<cphvb_int64>( instr, &score_less_equal );
                    break;
                case CPHVB_LESS_EQUAL*100+CPHVB_UINT8:
                    iter<cphvb_uint8>( instr, &score_less_equal );
                    break;
                case CPHVB_LESS_EQUAL*100+CPHVB_UINT16:
                    iter<cphvb_uint16>( instr, &score_less_equal );
                    break;
                case CPHVB_LESS_EQUAL*100+CPHVB_UINT32:
                    iter<cphvb_uint32>( instr, &score_less_equal );
                    break;
                case CPHVB_LESS_EQUAL*100+CPHVB_UINT64:
                    iter<cphvb_uint64>( instr, &score_less_equal );
                    break;
                case CPHVB_LESS_EQUAL*100+CPHVB_FLOAT32:
                    iter<cphvb_float32>( instr, &score_less_equal );
                    break;
                case CPHVB_LESS_EQUAL*100+CPHVB_FLOAT64:
                    iter<cphvb_float64>( instr, &score_less_equal );
                    break;
                case CPHVB_NOT_EQUAL*100+CPHVB_BOOL:
                    iter<cphvb_bool>( instr, &score_not_equal );
                    break;
                case CPHVB_NOT_EQUAL*100+CPHVB_INT8:
                    iter<cphvb_int8>( instr, &score_not_equal );
                    break;
                case CPHVB_NOT_EQUAL*100+CPHVB_INT16:
                    iter<cphvb_int16>( instr, &score_not_equal );
                    break;
                case CPHVB_NOT_EQUAL*100+CPHVB_INT32:
                    iter<cphvb_int32>( instr, &score_not_equal );
                    break;
                case CPHVB_NOT_EQUAL*100+CPHVB_INT64:
                    iter<cphvb_int64>( instr, &score_not_equal );
                    break;
                case CPHVB_NOT_EQUAL*100+CPHVB_UINT8:
                    iter<cphvb_uint8>( instr, &score_not_equal );
                    break;
                case CPHVB_NOT_EQUAL*100+CPHVB_UINT16:
                    iter<cphvb_uint16>( instr, &score_not_equal );
                    break;
                case CPHVB_NOT_EQUAL*100+CPHVB_UINT32:
                    iter<cphvb_uint32>( instr, &score_not_equal );
                    break;
                case CPHVB_NOT_EQUAL*100+CPHVB_UINT64:
                    iter<cphvb_uint64>( instr, &score_not_equal );
                    break;
                case CPHVB_NOT_EQUAL*100+CPHVB_FLOAT32:
                    iter<cphvb_float32>( instr, &score_not_equal );
                    break;
                case CPHVB_NOT_EQUAL*100+CPHVB_FLOAT64:
                    iter<cphvb_float64>( instr, &score_not_equal );
                    break;
                case CPHVB_EQUAL*100+CPHVB_BOOL:
                    iter<cphvb_bool>( instr, &score_equal );
                    break;
                case CPHVB_EQUAL*100+CPHVB_INT8:
                    iter<cphvb_int8>( instr, &score_equal );
                    break;
                case CPHVB_EQUAL*100+CPHVB_INT16:
                    iter<cphvb_int16>( instr, &score_equal );
                    break;
                case CPHVB_EQUAL*100+CPHVB_INT32:
                    iter<cphvb_int32>( instr, &score_equal );
                    break;
                case CPHVB_EQUAL*100+CPHVB_INT64:
                    iter<cphvb_int64>( instr, &score_equal );
                    break;
                case CPHVB_EQUAL*100+CPHVB_UINT8:
                    iter<cphvb_uint8>( instr, &score_equal );
                    break;
                case CPHVB_EQUAL*100+CPHVB_UINT16:
                    iter<cphvb_uint16>( instr, &score_equal );
                    break;
                case CPHVB_EQUAL*100+CPHVB_UINT32:
                    iter<cphvb_uint32>( instr, &score_equal );
                    break;
                case CPHVB_EQUAL*100+CPHVB_UINT64:
                    iter<cphvb_uint64>( instr, &score_equal );
                    break;
                case CPHVB_EQUAL*100+CPHVB_FLOAT32:
                    iter<cphvb_float32>( instr, &score_equal );
                    break;
                case CPHVB_EQUAL*100+CPHVB_FLOAT64:
                    iter<cphvb_float64>( instr, &score_equal );
                    break;
                case CPHVB_MAXIMUM*100+CPHVB_BOOL:
                    iter<cphvb_bool>( instr, &score_maximum );
                    break;
                case CPHVB_MAXIMUM*100+CPHVB_INT8:
                    iter<cphvb_int8>( instr, &score_maximum );
                    break;
                case CPHVB_MAXIMUM*100+CPHVB_INT16:
                    iter<cphvb_int16>( instr, &score_maximum );
                    break;
                case CPHVB_MAXIMUM*100+CPHVB_INT32:
                    iter<cphvb_int32>( instr, &score_maximum );
                    break;
                case CPHVB_MAXIMUM*100+CPHVB_INT64:
                    iter<cphvb_int64>( instr, &score_maximum );
                    break;
                case CPHVB_MAXIMUM*100+CPHVB_UINT8:
                    iter<cphvb_uint8>( instr, &score_maximum );
                    break;
                case CPHVB_MAXIMUM*100+CPHVB_UINT16:
                    iter<cphvb_uint16>( instr, &score_maximum );
                    break;
                case CPHVB_MAXIMUM*100+CPHVB_UINT32:
                    iter<cphvb_uint32>( instr, &score_maximum );
                    break;
                case CPHVB_MAXIMUM*100+CPHVB_UINT64:
                    iter<cphvb_uint64>( instr, &score_maximum );
                    break;
                case CPHVB_MAXIMUM*100+CPHVB_FLOAT32:
                    iter<cphvb_float32>( instr, &score_maximum );
                    break;
                case CPHVB_MAXIMUM*100+CPHVB_FLOAT64:
                    iter<cphvb_float64>( instr, &score_maximum );
                    break;
                case CPHVB_MINIMUM*100+CPHVB_BOOL:
                    iter<cphvb_bool>( instr, &score_minimum );
                    break;
                case CPHVB_MINIMUM*100+CPHVB_INT8:
                    iter<cphvb_int8>( instr, &score_minimum );
                    break;
                case CPHVB_MINIMUM*100+CPHVB_INT16:
                    iter<cphvb_int16>( instr, &score_minimum );
                    break;
                case CPHVB_MINIMUM*100+CPHVB_INT32:
                    iter<cphvb_int32>( instr, &score_minimum );
                    break;
                case CPHVB_MINIMUM*100+CPHVB_INT64:
                    iter<cphvb_int64>( instr, &score_minimum );
                    break;
                case CPHVB_MINIMUM*100+CPHVB_UINT8:
                    iter<cphvb_uint8>( instr, &score_minimum );
                    break;
                case CPHVB_MINIMUM*100+CPHVB_UINT16:
                    iter<cphvb_uint16>( instr, &score_minimum );
                    break;
                case CPHVB_MINIMUM*100+CPHVB_UINT32:
                    iter<cphvb_uint32>( instr, &score_minimum );
                    break;
                case CPHVB_MINIMUM*100+CPHVB_UINT64:
                    iter<cphvb_uint64>( instr, &score_minimum );
                    break;
                case CPHVB_MINIMUM*100+CPHVB_FLOAT32:
                    iter<cphvb_float32>( instr, &score_minimum );
                    break;
                case CPHVB_MINIMUM*100+CPHVB_FLOAT64:
                    iter<cphvb_float64>( instr, &score_minimum );
                    break;
                case CPHVB_LDEXP*100+CPHVB_BOOL:
                    iter<cphvb_bool>( instr, &score_ldexp );
                    break;
                case CPHVB_LDEXP*100+CPHVB_INT8:
                    iter<cphvb_int8>( instr, &score_ldexp );
                    break;
                case CPHVB_LDEXP*100+CPHVB_INT16:
                    iter<cphvb_int16>( instr, &score_ldexp );
                    break;
                case CPHVB_LDEXP*100+CPHVB_INT32:
                    iter<cphvb_int32>( instr, &score_ldexp );
                    break;
                case CPHVB_LDEXP*100+CPHVB_INT64:
                    iter<cphvb_int64>( instr, &score_ldexp );
                    break;
                case CPHVB_LDEXP*100+CPHVB_UINT8:
                    iter<cphvb_uint8>( instr, &score_ldexp );
                    break;
                case CPHVB_LDEXP*100+CPHVB_UINT16:
                    iter<cphvb_uint16>( instr, &score_ldexp );
                    break;
                case CPHVB_LDEXP*100+CPHVB_UINT32:
                    iter<cphvb_uint32>( instr, &score_ldexp );
                    break;
                case CPHVB_LDEXP*100+CPHVB_UINT64:
                    iter<cphvb_uint64>( instr, &score_ldexp );
                    break;
                case CPHVB_LDEXP*100+CPHVB_FLOAT32:
                    iter<cphvb_float32>( instr, &score_ldexp );
                    break;
                case CPHVB_LDEXP*100+CPHVB_FLOAT64:
                    iter<cphvb_float64>( instr, &score_ldexp );
                    break;
                case CPHVB_NEGATIVE*100+CPHVB_BOOL:
                    iter<cphvb_bool>( instr, &score_negative );
                    break;
                case CPHVB_NEGATIVE*100+CPHVB_INT8:
                    iter<cphvb_int8>( instr, &score_negative );
                    break;
                case CPHVB_NEGATIVE*100+CPHVB_INT16:
                    iter<cphvb_int16>( instr, &score_negative );
                    break;
                case CPHVB_NEGATIVE*100+CPHVB_INT32:
                    iter<cphvb_int32>( instr, &score_negative );
                    break;
                case CPHVB_NEGATIVE*100+CPHVB_INT64:
                    iter<cphvb_int64>( instr, &score_negative );
                    break;
                case CPHVB_NEGATIVE*100+CPHVB_UINT8:
                    iter<cphvb_uint8>( instr, &score_negative );
                    break;
                case CPHVB_NEGATIVE*100+CPHVB_UINT16:
                    iter<cphvb_uint16>( instr, &score_negative );
                    break;
                case CPHVB_NEGATIVE*100+CPHVB_UINT32:
                    iter<cphvb_uint32>( instr, &score_negative );
                    break;
                case CPHVB_NEGATIVE*100+CPHVB_UINT64:
                    iter<cphvb_uint64>( instr, &score_negative );
                    break;
                case CPHVB_NEGATIVE*100+CPHVB_FLOAT32:
                    iter<cphvb_float32>( instr, &score_negative );
                    break;
                case CPHVB_NEGATIVE*100+CPHVB_FLOAT64:
                    iter<cphvb_float64>( instr, &score_negative );
                    break;
                case CPHVB_ABSOLUTE*100+CPHVB_BOOL:
                    iter<cphvb_bool>( instr, &score_absolute );
                    break;
                case CPHVB_ABSOLUTE*100+CPHVB_INT8:
                    iter<cphvb_int8>( instr, &score_absolute );
                    break;
                case CPHVB_ABSOLUTE*100+CPHVB_INT16:
                    iter<cphvb_int16>( instr, &score_absolute );
                    break;
                case CPHVB_ABSOLUTE*100+CPHVB_INT32:
                    iter<cphvb_int32>( instr, &score_absolute );
                    break;
                case CPHVB_ABSOLUTE*100+CPHVB_INT64:
                    iter<cphvb_int64>( instr, &score_absolute );
                    break;
                case CPHVB_ABSOLUTE*100+CPHVB_UINT8:
                    iter<cphvb_uint8>( instr, &score_absolute );
                    break;
                case CPHVB_ABSOLUTE*100+CPHVB_UINT16:
                    iter<cphvb_uint16>( instr, &score_absolute );
                    break;
                case CPHVB_ABSOLUTE*100+CPHVB_UINT32:
                    iter<cphvb_uint32>( instr, &score_absolute );
                    break;
                case CPHVB_ABSOLUTE*100+CPHVB_UINT64:
                    iter<cphvb_uint64>( instr, &score_absolute );
                    break;
                case CPHVB_ABSOLUTE*100+CPHVB_FLOAT32:
                    iter<cphvb_float32>( instr, &score_absolute );
                    break;
                case CPHVB_ABSOLUTE*100+CPHVB_FLOAT64:
                    iter<cphvb_float64>( instr, &score_absolute );
                    break;
                case CPHVB_RINT*100+CPHVB_BOOL:
                    iter<cphvb_bool>( instr, &score_rint );
                    break;
                case CPHVB_RINT*100+CPHVB_INT8:
                    iter<cphvb_int8>( instr, &score_rint );
                    break;
                case CPHVB_RINT*100+CPHVB_INT16:
                    iter<cphvb_int16>( instr, &score_rint );
                    break;
                case CPHVB_RINT*100+CPHVB_INT32:
                    iter<cphvb_int32>( instr, &score_rint );
                    break;
                case CPHVB_RINT*100+CPHVB_INT64:
                    iter<cphvb_int64>( instr, &score_rint );
                    break;
                case CPHVB_RINT*100+CPHVB_UINT8:
                    iter<cphvb_uint8>( instr, &score_rint );
                    break;
                case CPHVB_RINT*100+CPHVB_UINT16:
                    iter<cphvb_uint16>( instr, &score_rint );
                    break;
                case CPHVB_RINT*100+CPHVB_UINT32:
                    iter<cphvb_uint32>( instr, &score_rint );
                    break;
                case CPHVB_RINT*100+CPHVB_UINT64:
                    iter<cphvb_uint64>( instr, &score_rint );
                    break;
                case CPHVB_RINT*100+CPHVB_FLOAT32:
                    iter<cphvb_float32>( instr, &score_rint );
                    break;
                case CPHVB_RINT*100+CPHVB_FLOAT64:
                    iter<cphvb_float64>( instr, &score_rint );
                    break;
                case CPHVB_SIGN*100+CPHVB_BOOL:
                    iter<cphvb_bool>( instr, &score_sign );
                    break;
                case CPHVB_SIGN*100+CPHVB_INT8:
                    iter<cphvb_int8>( instr, &score_sign );
                    break;
                case CPHVB_SIGN*100+CPHVB_INT16:
                    iter<cphvb_int16>( instr, &score_sign );
                    break;
                case CPHVB_SIGN*100+CPHVB_INT32:
                    iter<cphvb_int32>( instr, &score_sign );
                    break;
                case CPHVB_SIGN*100+CPHVB_INT64:
                    iter<cphvb_int64>( instr, &score_sign );
                    break;
                case CPHVB_SIGN*100+CPHVB_UINT8:
                    iter<cphvb_uint8>( instr, &score_sign );
                    break;
                case CPHVB_SIGN*100+CPHVB_UINT16:
                    iter<cphvb_uint16>( instr, &score_sign );
                    break;
                case CPHVB_SIGN*100+CPHVB_UINT32:
                    iter<cphvb_uint32>( instr, &score_sign );
                    break;
                case CPHVB_SIGN*100+CPHVB_UINT64:
                    iter<cphvb_uint64>( instr, &score_sign );
                    break;
                case CPHVB_SIGN*100+CPHVB_FLOAT32:
                    iter<cphvb_float32>( instr, &score_sign );
                    break;
                case CPHVB_SIGN*100+CPHVB_FLOAT64:
                    iter<cphvb_float64>( instr, &score_sign );
                    break;
                case CPHVB_CONJ*100+CPHVB_BOOL:
                    iter<cphvb_bool>( instr, &score_conj );
                    break;
                case CPHVB_CONJ*100+CPHVB_INT8:
                    iter<cphvb_int8>( instr, &score_conj );
                    break;
                case CPHVB_CONJ*100+CPHVB_INT16:
                    iter<cphvb_int16>( instr, &score_conj );
                    break;
                case CPHVB_CONJ*100+CPHVB_INT32:
                    iter<cphvb_int32>( instr, &score_conj );
                    break;
                case CPHVB_CONJ*100+CPHVB_INT64:
                    iter<cphvb_int64>( instr, &score_conj );
                    break;
                case CPHVB_CONJ*100+CPHVB_UINT8:
                    iter<cphvb_uint8>( instr, &score_conj );
                    break;
                case CPHVB_CONJ*100+CPHVB_UINT16:
                    iter<cphvb_uint16>( instr, &score_conj );
                    break;
                case CPHVB_CONJ*100+CPHVB_UINT32:
                    iter<cphvb_uint32>( instr, &score_conj );
                    break;
                case CPHVB_CONJ*100+CPHVB_UINT64:
                    iter<cphvb_uint64>( instr, &score_conj );
                    break;
                case CPHVB_CONJ*100+CPHVB_FLOAT32:
                    iter<cphvb_float32>( instr, &score_conj );
                    break;
                case CPHVB_CONJ*100+CPHVB_FLOAT64:
                    iter<cphvb_float64>( instr, &score_conj );
                    break;
                case CPHVB_EXP*100+CPHVB_BOOL:
                    iter<cphvb_bool>( instr, &score_exp );
                    break;
                case CPHVB_EXP*100+CPHVB_INT8:
                    iter<cphvb_int8>( instr, &score_exp );
                    break;
                case CPHVB_EXP*100+CPHVB_INT16:
                    iter<cphvb_int16>( instr, &score_exp );
                    break;
                case CPHVB_EXP*100+CPHVB_INT32:
                    iter<cphvb_int32>( instr, &score_exp );
                    break;
                case CPHVB_EXP*100+CPHVB_INT64:
                    iter<cphvb_int64>( instr, &score_exp );
                    break;
                case CPHVB_EXP*100+CPHVB_UINT8:
                    iter<cphvb_uint8>( instr, &score_exp );
                    break;
                case CPHVB_EXP*100+CPHVB_UINT16:
                    iter<cphvb_uint16>( instr, &score_exp );
                    break;
                case CPHVB_EXP*100+CPHVB_UINT32:
                    iter<cphvb_uint32>( instr, &score_exp );
                    break;
                case CPHVB_EXP*100+CPHVB_UINT64:
                    iter<cphvb_uint64>( instr, &score_exp );
                    break;
                case CPHVB_EXP*100+CPHVB_FLOAT32:
                    iter<cphvb_float32>( instr, &score_exp );
                    break;
                case CPHVB_EXP*100+CPHVB_FLOAT64:
                    iter<cphvb_float64>( instr, &score_exp );
                    break;
                case CPHVB_EXP2*100+CPHVB_BOOL:
                    iter<cphvb_bool>( instr, &score_exp2 );
                    break;
                case CPHVB_EXP2*100+CPHVB_INT8:
                    iter<cphvb_int8>( instr, &score_exp2 );
                    break;
                case CPHVB_EXP2*100+CPHVB_INT16:
                    iter<cphvb_int16>( instr, &score_exp2 );
                    break;
                case CPHVB_EXP2*100+CPHVB_INT32:
                    iter<cphvb_int32>( instr, &score_exp2 );
                    break;
                case CPHVB_EXP2*100+CPHVB_INT64:
                    iter<cphvb_int64>( instr, &score_exp2 );
                    break;
                case CPHVB_EXP2*100+CPHVB_UINT8:
                    iter<cphvb_uint8>( instr, &score_exp2 );
                    break;
                case CPHVB_EXP2*100+CPHVB_UINT16:
                    iter<cphvb_uint16>( instr, &score_exp2 );
                    break;
                case CPHVB_EXP2*100+CPHVB_UINT32:
                    iter<cphvb_uint32>( instr, &score_exp2 );
                    break;
                case CPHVB_EXP2*100+CPHVB_UINT64:
                    iter<cphvb_uint64>( instr, &score_exp2 );
                    break;
                case CPHVB_EXP2*100+CPHVB_FLOAT32:
                    iter<cphvb_float32>( instr, &score_exp2 );
                    break;
                case CPHVB_EXP2*100+CPHVB_FLOAT64:
                    iter<cphvb_float64>( instr, &score_exp2 );
                    break;
                case CPHVB_LOG2*100+CPHVB_BOOL:
                    iter<cphvb_bool>( instr, &score_log2 );
                    break;
                case CPHVB_LOG2*100+CPHVB_INT8:
                    iter<cphvb_int8>( instr, &score_log2 );
                    break;
                case CPHVB_LOG2*100+CPHVB_INT16:
                    iter<cphvb_int16>( instr, &score_log2 );
                    break;
                case CPHVB_LOG2*100+CPHVB_INT32:
                    iter<cphvb_int32>( instr, &score_log2 );
                    break;
                case CPHVB_LOG2*100+CPHVB_INT64:
                    iter<cphvb_int64>( instr, &score_log2 );
                    break;
                case CPHVB_LOG2*100+CPHVB_UINT8:
                    iter<cphvb_uint8>( instr, &score_log2 );
                    break;
                case CPHVB_LOG2*100+CPHVB_UINT16:
                    iter<cphvb_uint16>( instr, &score_log2 );
                    break;
                case CPHVB_LOG2*100+CPHVB_UINT32:
                    iter<cphvb_uint32>( instr, &score_log2 );
                    break;
                case CPHVB_LOG2*100+CPHVB_UINT64:
                    iter<cphvb_uint64>( instr, &score_log2 );
                    break;
                case CPHVB_LOG2*100+CPHVB_FLOAT32:
                    iter<cphvb_float32>( instr, &score_log2 );
                    break;
                case CPHVB_LOG2*100+CPHVB_FLOAT64:
                    iter<cphvb_float64>( instr, &score_log2 );
                    break;
                case CPHVB_LOG*100+CPHVB_BOOL:
                    iter<cphvb_bool>( instr, &score_log );
                    break;
                case CPHVB_LOG*100+CPHVB_INT8:
                    iter<cphvb_int8>( instr, &score_log );
                    break;
                case CPHVB_LOG*100+CPHVB_INT16:
                    iter<cphvb_int16>( instr, &score_log );
                    break;
                case CPHVB_LOG*100+CPHVB_INT32:
                    iter<cphvb_int32>( instr, &score_log );
                    break;
                case CPHVB_LOG*100+CPHVB_INT64:
                    iter<cphvb_int64>( instr, &score_log );
                    break;
                case CPHVB_LOG*100+CPHVB_UINT8:
                    iter<cphvb_uint8>( instr, &score_log );
                    break;
                case CPHVB_LOG*100+CPHVB_UINT16:
                    iter<cphvb_uint16>( instr, &score_log );
                    break;
                case CPHVB_LOG*100+CPHVB_UINT32:
                    iter<cphvb_uint32>( instr, &score_log );
                    break;
                case CPHVB_LOG*100+CPHVB_UINT64:
                    iter<cphvb_uint64>( instr, &score_log );
                    break;
                case CPHVB_LOG*100+CPHVB_FLOAT32:
                    iter<cphvb_float32>( instr, &score_log );
                    break;
                case CPHVB_LOG*100+CPHVB_FLOAT64:
                    iter<cphvb_float64>( instr, &score_log );
                    break;
                case CPHVB_LOG10*100+CPHVB_BOOL:
                    iter<cphvb_bool>( instr, &score_log10 );
                    break;
                case CPHVB_LOG10*100+CPHVB_INT8:
                    iter<cphvb_int8>( instr, &score_log10 );
                    break;
                case CPHVB_LOG10*100+CPHVB_INT16:
                    iter<cphvb_int16>( instr, &score_log10 );
                    break;
                case CPHVB_LOG10*100+CPHVB_INT32:
                    iter<cphvb_int32>( instr, &score_log10 );
                    break;
                case CPHVB_LOG10*100+CPHVB_INT64:
                    iter<cphvb_int64>( instr, &score_log10 );
                    break;
                case CPHVB_LOG10*100+CPHVB_UINT8:
                    iter<cphvb_uint8>( instr, &score_log10 );
                    break;
                case CPHVB_LOG10*100+CPHVB_UINT16:
                    iter<cphvb_uint16>( instr, &score_log10 );
                    break;
                case CPHVB_LOG10*100+CPHVB_UINT32:
                    iter<cphvb_uint32>( instr, &score_log10 );
                    break;
                case CPHVB_LOG10*100+CPHVB_UINT64:
                    iter<cphvb_uint64>( instr, &score_log10 );
                    break;
                case CPHVB_LOG10*100+CPHVB_FLOAT32:
                    iter<cphvb_float32>( instr, &score_log10 );
                    break;
                case CPHVB_LOG10*100+CPHVB_FLOAT64:
                    iter<cphvb_float64>( instr, &score_log10 );
                    break;
                case CPHVB_EXPM1*100+CPHVB_BOOL:
                    iter<cphvb_bool>( instr, &score_expm1 );
                    break;
                case CPHVB_EXPM1*100+CPHVB_INT8:
                    iter<cphvb_int8>( instr, &score_expm1 );
                    break;
                case CPHVB_EXPM1*100+CPHVB_INT16:
                    iter<cphvb_int16>( instr, &score_expm1 );
                    break;
                case CPHVB_EXPM1*100+CPHVB_INT32:
                    iter<cphvb_int32>( instr, &score_expm1 );
                    break;
                case CPHVB_EXPM1*100+CPHVB_INT64:
                    iter<cphvb_int64>( instr, &score_expm1 );
                    break;
                case CPHVB_EXPM1*100+CPHVB_UINT8:
                    iter<cphvb_uint8>( instr, &score_expm1 );
                    break;
                case CPHVB_EXPM1*100+CPHVB_UINT16:
                    iter<cphvb_uint16>( instr, &score_expm1 );
                    break;
                case CPHVB_EXPM1*100+CPHVB_UINT32:
                    iter<cphvb_uint32>( instr, &score_expm1 );
                    break;
                case CPHVB_EXPM1*100+CPHVB_UINT64:
                    iter<cphvb_uint64>( instr, &score_expm1 );
                    break;
                case CPHVB_EXPM1*100+CPHVB_FLOAT32:
                    iter<cphvb_float32>( instr, &score_expm1 );
                    break;
                case CPHVB_EXPM1*100+CPHVB_FLOAT64:
                    iter<cphvb_float64>( instr, &score_expm1 );
                    break;
                case CPHVB_LOG1P*100+CPHVB_BOOL:
                    iter<cphvb_bool>( instr, &score_log1p );
                    break;
                case CPHVB_LOG1P*100+CPHVB_INT8:
                    iter<cphvb_int8>( instr, &score_log1p );
                    break;
                case CPHVB_LOG1P*100+CPHVB_INT16:
                    iter<cphvb_int16>( instr, &score_log1p );
                    break;
                case CPHVB_LOG1P*100+CPHVB_INT32:
                    iter<cphvb_int32>( instr, &score_log1p );
                    break;
                case CPHVB_LOG1P*100+CPHVB_INT64:
                    iter<cphvb_int64>( instr, &score_log1p );
                    break;
                case CPHVB_LOG1P*100+CPHVB_UINT8:
                    iter<cphvb_uint8>( instr, &score_log1p );
                    break;
                case CPHVB_LOG1P*100+CPHVB_UINT16:
                    iter<cphvb_uint16>( instr, &score_log1p );
                    break;
                case CPHVB_LOG1P*100+CPHVB_UINT32:
                    iter<cphvb_uint32>( instr, &score_log1p );
                    break;
                case CPHVB_LOG1P*100+CPHVB_UINT64:
                    iter<cphvb_uint64>( instr, &score_log1p );
                    break;
                case CPHVB_LOG1P*100+CPHVB_FLOAT32:
                    iter<cphvb_float32>( instr, &score_log1p );
                    break;
                case CPHVB_LOG1P*100+CPHVB_FLOAT64:
                    iter<cphvb_float64>( instr, &score_log1p );
                    break;
                case CPHVB_SQRT*100+CPHVB_BOOL:
                    iter<cphvb_bool>( instr, &score_sqrt );
                    break;
                case CPHVB_SQRT*100+CPHVB_INT8:
                    iter<cphvb_int8>( instr, &score_sqrt );
                    break;
                case CPHVB_SQRT*100+CPHVB_INT16:
                    iter<cphvb_int16>( instr, &score_sqrt );
                    break;
                case CPHVB_SQRT*100+CPHVB_INT32:
                    iter<cphvb_int32>( instr, &score_sqrt );
                    break;
                case CPHVB_SQRT*100+CPHVB_INT64:
                    iter<cphvb_int64>( instr, &score_sqrt );
                    break;
                case CPHVB_SQRT*100+CPHVB_UINT8:
                    iter<cphvb_uint8>( instr, &score_sqrt );
                    break;
                case CPHVB_SQRT*100+CPHVB_UINT16:
                    iter<cphvb_uint16>( instr, &score_sqrt );
                    break;
                case CPHVB_SQRT*100+CPHVB_UINT32:
                    iter<cphvb_uint32>( instr, &score_sqrt );
                    break;
                case CPHVB_SQRT*100+CPHVB_UINT64:
                    iter<cphvb_uint64>( instr, &score_sqrt );
                    break;
                case CPHVB_SQRT*100+CPHVB_FLOAT32:
                    iter<cphvb_float32>( instr, &score_sqrt );
                    break;
                case CPHVB_SQRT*100+CPHVB_FLOAT64:
                    iter<cphvb_float64>( instr, &score_sqrt );
                    break;
                case CPHVB_SQUARE*100+CPHVB_BOOL:
                    iter<cphvb_bool>( instr, &score_square );
                    break;
                case CPHVB_SQUARE*100+CPHVB_INT8:
                    iter<cphvb_int8>( instr, &score_square );
                    break;
                case CPHVB_SQUARE*100+CPHVB_INT16:
                    iter<cphvb_int16>( instr, &score_square );
                    break;
                case CPHVB_SQUARE*100+CPHVB_INT32:
                    iter<cphvb_int32>( instr, &score_square );
                    break;
                case CPHVB_SQUARE*100+CPHVB_INT64:
                    iter<cphvb_int64>( instr, &score_square );
                    break;
                case CPHVB_SQUARE*100+CPHVB_UINT8:
                    iter<cphvb_uint8>( instr, &score_square );
                    break;
                case CPHVB_SQUARE*100+CPHVB_UINT16:
                    iter<cphvb_uint16>( instr, &score_square );
                    break;
                case CPHVB_SQUARE*100+CPHVB_UINT32:
                    iter<cphvb_uint32>( instr, &score_square );
                    break;
                case CPHVB_SQUARE*100+CPHVB_UINT64:
                    iter<cphvb_uint64>( instr, &score_square );
                    break;
                case CPHVB_SQUARE*100+CPHVB_FLOAT32:
                    iter<cphvb_float32>( instr, &score_square );
                    break;
                case CPHVB_SQUARE*100+CPHVB_FLOAT64:
                    iter<cphvb_float64>( instr, &score_square );
                    break;
                case CPHVB_RECIPROCAL*100+CPHVB_BOOL:
                    iter<cphvb_bool>( instr, &score_reciprocal );
                    break;
                case CPHVB_RECIPROCAL*100+CPHVB_INT8:
                    iter<cphvb_int8>( instr, &score_reciprocal );
                    break;
                case CPHVB_RECIPROCAL*100+CPHVB_INT16:
                    iter<cphvb_int16>( instr, &score_reciprocal );
                    break;
                case CPHVB_RECIPROCAL*100+CPHVB_INT32:
                    iter<cphvb_int32>( instr, &score_reciprocal );
                    break;
                case CPHVB_RECIPROCAL*100+CPHVB_INT64:
                    iter<cphvb_int64>( instr, &score_reciprocal );
                    break;
                case CPHVB_RECIPROCAL*100+CPHVB_UINT8:
                    iter<cphvb_uint8>( instr, &score_reciprocal );
                    break;
                case CPHVB_RECIPROCAL*100+CPHVB_UINT16:
                    iter<cphvb_uint16>( instr, &score_reciprocal );
                    break;
                case CPHVB_RECIPROCAL*100+CPHVB_UINT32:
                    iter<cphvb_uint32>( instr, &score_reciprocal );
                    break;
                case CPHVB_RECIPROCAL*100+CPHVB_UINT64:
                    iter<cphvb_uint64>( instr, &score_reciprocal );
                    break;
                case CPHVB_RECIPROCAL*100+CPHVB_FLOAT32:
                    iter<cphvb_float32>( instr, &score_reciprocal );
                    break;
                case CPHVB_RECIPROCAL*100+CPHVB_FLOAT64:
                    iter<cphvb_float64>( instr, &score_reciprocal );
                    break;
                case CPHVB_ONES_LIKE*100+CPHVB_BOOL:
                    iter<cphvb_bool>( instr, &score_ones_like );
                    break;
                case CPHVB_ONES_LIKE*100+CPHVB_INT8:
                    iter<cphvb_int8>( instr, &score_ones_like );
                    break;
                case CPHVB_ONES_LIKE*100+CPHVB_INT16:
                    iter<cphvb_int16>( instr, &score_ones_like );
                    break;
                case CPHVB_ONES_LIKE*100+CPHVB_INT32:
                    iter<cphvb_int32>( instr, &score_ones_like );
                    break;
                case CPHVB_ONES_LIKE*100+CPHVB_INT64:
                    iter<cphvb_int64>( instr, &score_ones_like );
                    break;
                case CPHVB_ONES_LIKE*100+CPHVB_UINT8:
                    iter<cphvb_uint8>( instr, &score_ones_like );
                    break;
                case CPHVB_ONES_LIKE*100+CPHVB_UINT16:
                    iter<cphvb_uint16>( instr, &score_ones_like );
                    break;
                case CPHVB_ONES_LIKE*100+CPHVB_UINT32:
                    iter<cphvb_uint32>( instr, &score_ones_like );
                    break;
                case CPHVB_ONES_LIKE*100+CPHVB_UINT64:
                    iter<cphvb_uint64>( instr, &score_ones_like );
                    break;
                case CPHVB_ONES_LIKE*100+CPHVB_FLOAT32:
                    iter<cphvb_float32>( instr, &score_ones_like );
                    break;
                case CPHVB_ONES_LIKE*100+CPHVB_FLOAT64:
                    iter<cphvb_float64>( instr, &score_ones_like );
                    break;
                case CPHVB_SIN*100+CPHVB_BOOL:
                    iter<cphvb_bool>( instr, &score_sin );
                    break;
                case CPHVB_SIN*100+CPHVB_INT8:
                    iter<cphvb_int8>( instr, &score_sin );
                    break;
                case CPHVB_SIN*100+CPHVB_INT16:
                    iter<cphvb_int16>( instr, &score_sin );
                    break;
                case CPHVB_SIN*100+CPHVB_INT32:
                    iter<cphvb_int32>( instr, &score_sin );
                    break;
                case CPHVB_SIN*100+CPHVB_INT64:
                    iter<cphvb_int64>( instr, &score_sin );
                    break;
                case CPHVB_SIN*100+CPHVB_UINT8:
                    iter<cphvb_uint8>( instr, &score_sin );
                    break;
                case CPHVB_SIN*100+CPHVB_UINT16:
                    iter<cphvb_uint16>( instr, &score_sin );
                    break;
                case CPHVB_SIN*100+CPHVB_UINT32:
                    iter<cphvb_uint32>( instr, &score_sin );
                    break;
                case CPHVB_SIN*100+CPHVB_UINT64:
                    iter<cphvb_uint64>( instr, &score_sin );
                    break;
                case CPHVB_SIN*100+CPHVB_FLOAT32:
                    iter<cphvb_float32>( instr, &score_sin );
                    break;
                case CPHVB_SIN*100+CPHVB_FLOAT64:
                    iter<cphvb_float64>( instr, &score_sin );
                    break;
                case CPHVB_COS*100+CPHVB_BOOL:
                    iter<cphvb_bool>( instr, &score_cos );
                    break;
                case CPHVB_COS*100+CPHVB_INT8:
                    iter<cphvb_int8>( instr, &score_cos );
                    break;
                case CPHVB_COS*100+CPHVB_INT16:
                    iter<cphvb_int16>( instr, &score_cos );
                    break;
                case CPHVB_COS*100+CPHVB_INT32:
                    iter<cphvb_int32>( instr, &score_cos );
                    break;
                case CPHVB_COS*100+CPHVB_INT64:
                    iter<cphvb_int64>( instr, &score_cos );
                    break;
                case CPHVB_COS*100+CPHVB_UINT8:
                    iter<cphvb_uint8>( instr, &score_cos );
                    break;
                case CPHVB_COS*100+CPHVB_UINT16:
                    iter<cphvb_uint16>( instr, &score_cos );
                    break;
                case CPHVB_COS*100+CPHVB_UINT32:
                    iter<cphvb_uint32>( instr, &score_cos );
                    break;
                case CPHVB_COS*100+CPHVB_UINT64:
                    iter<cphvb_uint64>( instr, &score_cos );
                    break;
                case CPHVB_COS*100+CPHVB_FLOAT32:
                    iter<cphvb_float32>( instr, &score_cos );
                    break;
                case CPHVB_COS*100+CPHVB_FLOAT64:
                    iter<cphvb_float64>( instr, &score_cos );
                    break;
                case CPHVB_TAN*100+CPHVB_BOOL:
                    iter<cphvb_bool>( instr, &score_tan );
                    break;
                case CPHVB_TAN*100+CPHVB_INT8:
                    iter<cphvb_int8>( instr, &score_tan );
                    break;
                case CPHVB_TAN*100+CPHVB_INT16:
                    iter<cphvb_int16>( instr, &score_tan );
                    break;
                case CPHVB_TAN*100+CPHVB_INT32:
                    iter<cphvb_int32>( instr, &score_tan );
                    break;
                case CPHVB_TAN*100+CPHVB_INT64:
                    iter<cphvb_int64>( instr, &score_tan );
                    break;
                case CPHVB_TAN*100+CPHVB_UINT8:
                    iter<cphvb_uint8>( instr, &score_tan );
                    break;
                case CPHVB_TAN*100+CPHVB_UINT16:
                    iter<cphvb_uint16>( instr, &score_tan );
                    break;
                case CPHVB_TAN*100+CPHVB_UINT32:
                    iter<cphvb_uint32>( instr, &score_tan );
                    break;
                case CPHVB_TAN*100+CPHVB_UINT64:
                    iter<cphvb_uint64>( instr, &score_tan );
                    break;
                case CPHVB_TAN*100+CPHVB_FLOAT32:
                    iter<cphvb_float32>( instr, &score_tan );
                    break;
                case CPHVB_TAN*100+CPHVB_FLOAT64:
                    iter<cphvb_float64>( instr, &score_tan );
                    break;
                case CPHVB_ARCSIN*100+CPHVB_BOOL:
                    iter<cphvb_bool>( instr, &score_arcsin );
                    break;
                case CPHVB_ARCSIN*100+CPHVB_INT8:
                    iter<cphvb_int8>( instr, &score_arcsin );
                    break;
                case CPHVB_ARCSIN*100+CPHVB_INT16:
                    iter<cphvb_int16>( instr, &score_arcsin );
                    break;
                case CPHVB_ARCSIN*100+CPHVB_INT32:
                    iter<cphvb_int32>( instr, &score_arcsin );
                    break;
                case CPHVB_ARCSIN*100+CPHVB_INT64:
                    iter<cphvb_int64>( instr, &score_arcsin );
                    break;
                case CPHVB_ARCSIN*100+CPHVB_UINT8:
                    iter<cphvb_uint8>( instr, &score_arcsin );
                    break;
                case CPHVB_ARCSIN*100+CPHVB_UINT16:
                    iter<cphvb_uint16>( instr, &score_arcsin );
                    break;
                case CPHVB_ARCSIN*100+CPHVB_UINT32:
                    iter<cphvb_uint32>( instr, &score_arcsin );
                    break;
                case CPHVB_ARCSIN*100+CPHVB_UINT64:
                    iter<cphvb_uint64>( instr, &score_arcsin );
                    break;
                case CPHVB_ARCSIN*100+CPHVB_FLOAT32:
                    iter<cphvb_float32>( instr, &score_arcsin );
                    break;
                case CPHVB_ARCSIN*100+CPHVB_FLOAT64:
                    iter<cphvb_float64>( instr, &score_arcsin );
                    break;
                case CPHVB_ARCCOS*100+CPHVB_BOOL:
                    iter<cphvb_bool>( instr, &score_arccos );
                    break;
                case CPHVB_ARCCOS*100+CPHVB_INT8:
                    iter<cphvb_int8>( instr, &score_arccos );
                    break;
                case CPHVB_ARCCOS*100+CPHVB_INT16:
                    iter<cphvb_int16>( instr, &score_arccos );
                    break;
                case CPHVB_ARCCOS*100+CPHVB_INT32:
                    iter<cphvb_int32>( instr, &score_arccos );
                    break;
                case CPHVB_ARCCOS*100+CPHVB_INT64:
                    iter<cphvb_int64>( instr, &score_arccos );
                    break;
                case CPHVB_ARCCOS*100+CPHVB_UINT8:
                    iter<cphvb_uint8>( instr, &score_arccos );
                    break;
                case CPHVB_ARCCOS*100+CPHVB_UINT16:
                    iter<cphvb_uint16>( instr, &score_arccos );
                    break;
                case CPHVB_ARCCOS*100+CPHVB_UINT32:
                    iter<cphvb_uint32>( instr, &score_arccos );
                    break;
                case CPHVB_ARCCOS*100+CPHVB_UINT64:
                    iter<cphvb_uint64>( instr, &score_arccos );
                    break;
                case CPHVB_ARCCOS*100+CPHVB_FLOAT32:
                    iter<cphvb_float32>( instr, &score_arccos );
                    break;
                case CPHVB_ARCCOS*100+CPHVB_FLOAT64:
                    iter<cphvb_float64>( instr, &score_arccos );
                    break;
                case CPHVB_ARCTAN*100+CPHVB_BOOL:
                    iter<cphvb_bool>( instr, &score_arctan );
                    break;
                case CPHVB_ARCTAN*100+CPHVB_INT8:
                    iter<cphvb_int8>( instr, &score_arctan );
                    break;
                case CPHVB_ARCTAN*100+CPHVB_INT16:
                    iter<cphvb_int16>( instr, &score_arctan );
                    break;
                case CPHVB_ARCTAN*100+CPHVB_INT32:
                    iter<cphvb_int32>( instr, &score_arctan );
                    break;
                case CPHVB_ARCTAN*100+CPHVB_INT64:
                    iter<cphvb_int64>( instr, &score_arctan );
                    break;
                case CPHVB_ARCTAN*100+CPHVB_UINT8:
                    iter<cphvb_uint8>( instr, &score_arctan );
                    break;
                case CPHVB_ARCTAN*100+CPHVB_UINT16:
                    iter<cphvb_uint16>( instr, &score_arctan );
                    break;
                case CPHVB_ARCTAN*100+CPHVB_UINT32:
                    iter<cphvb_uint32>( instr, &score_arctan );
                    break;
                case CPHVB_ARCTAN*100+CPHVB_UINT64:
                    iter<cphvb_uint64>( instr, &score_arctan );
                    break;
                case CPHVB_ARCTAN*100+CPHVB_FLOAT32:
                    iter<cphvb_float32>( instr, &score_arctan );
                    break;
                case CPHVB_ARCTAN*100+CPHVB_FLOAT64:
                    iter<cphvb_float64>( instr, &score_arctan );
                    break;
                case CPHVB_ARCTAN2*100+CPHVB_BOOL:
                    iter<cphvb_bool>( instr, &score_arctan2 );
                    break;
                case CPHVB_ARCTAN2*100+CPHVB_INT8:
                    iter<cphvb_int8>( instr, &score_arctan2 );
                    break;
                case CPHVB_ARCTAN2*100+CPHVB_INT16:
                    iter<cphvb_int16>( instr, &score_arctan2 );
                    break;
                case CPHVB_ARCTAN2*100+CPHVB_INT32:
                    iter<cphvb_int32>( instr, &score_arctan2 );
                    break;
                case CPHVB_ARCTAN2*100+CPHVB_INT64:
                    iter<cphvb_int64>( instr, &score_arctan2 );
                    break;
                case CPHVB_ARCTAN2*100+CPHVB_UINT8:
                    iter<cphvb_uint8>( instr, &score_arctan2 );
                    break;
                case CPHVB_ARCTAN2*100+CPHVB_UINT16:
                    iter<cphvb_uint16>( instr, &score_arctan2 );
                    break;
                case CPHVB_ARCTAN2*100+CPHVB_UINT32:
                    iter<cphvb_uint32>( instr, &score_arctan2 );
                    break;
                case CPHVB_ARCTAN2*100+CPHVB_UINT64:
                    iter<cphvb_uint64>( instr, &score_arctan2 );
                    break;
                case CPHVB_ARCTAN2*100+CPHVB_FLOAT32:
                    iter<cphvb_float32>( instr, &score_arctan2 );
                    break;
                case CPHVB_ARCTAN2*100+CPHVB_FLOAT64:
                    iter<cphvb_float64>( instr, &score_arctan2 );
                    break;
                case CPHVB_HYPOT*100+CPHVB_BOOL:
                    iter<cphvb_bool>( instr, &score_hypot );
                    break;
                case CPHVB_HYPOT*100+CPHVB_INT8:
                    iter<cphvb_int8>( instr, &score_hypot );
                    break;
                case CPHVB_HYPOT*100+CPHVB_INT16:
                    iter<cphvb_int16>( instr, &score_hypot );
                    break;
                case CPHVB_HYPOT*100+CPHVB_INT32:
                    iter<cphvb_int32>( instr, &score_hypot );
                    break;
                case CPHVB_HYPOT*100+CPHVB_INT64:
                    iter<cphvb_int64>( instr, &score_hypot );
                    break;
                case CPHVB_HYPOT*100+CPHVB_UINT8:
                    iter<cphvb_uint8>( instr, &score_hypot );
                    break;
                case CPHVB_HYPOT*100+CPHVB_UINT16:
                    iter<cphvb_uint16>( instr, &score_hypot );
                    break;
                case CPHVB_HYPOT*100+CPHVB_UINT32:
                    iter<cphvb_uint32>( instr, &score_hypot );
                    break;
                case CPHVB_HYPOT*100+CPHVB_UINT64:
                    iter<cphvb_uint64>( instr, &score_hypot );
                    break;
                case CPHVB_HYPOT*100+CPHVB_FLOAT32:
                    iter<cphvb_float32>( instr, &score_hypot );
                    break;
                case CPHVB_HYPOT*100+CPHVB_FLOAT64:
                    iter<cphvb_float64>( instr, &score_hypot );
                    break;
                case CPHVB_SINH*100+CPHVB_BOOL:
                    iter<cphvb_bool>( instr, &score_sinh );
                    break;
                case CPHVB_SINH*100+CPHVB_INT8:
                    iter<cphvb_int8>( instr, &score_sinh );
                    break;
                case CPHVB_SINH*100+CPHVB_INT16:
                    iter<cphvb_int16>( instr, &score_sinh );
                    break;
                case CPHVB_SINH*100+CPHVB_INT32:
                    iter<cphvb_int32>( instr, &score_sinh );
                    break;
                case CPHVB_SINH*100+CPHVB_INT64:
                    iter<cphvb_int64>( instr, &score_sinh );
                    break;
                case CPHVB_SINH*100+CPHVB_UINT8:
                    iter<cphvb_uint8>( instr, &score_sinh );
                    break;
                case CPHVB_SINH*100+CPHVB_UINT16:
                    iter<cphvb_uint16>( instr, &score_sinh );
                    break;
                case CPHVB_SINH*100+CPHVB_UINT32:
                    iter<cphvb_uint32>( instr, &score_sinh );
                    break;
                case CPHVB_SINH*100+CPHVB_UINT64:
                    iter<cphvb_uint64>( instr, &score_sinh );
                    break;
                case CPHVB_SINH*100+CPHVB_FLOAT32:
                    iter<cphvb_float32>( instr, &score_sinh );
                    break;
                case CPHVB_SINH*100+CPHVB_FLOAT64:
                    iter<cphvb_float64>( instr, &score_sinh );
                    break;
                case CPHVB_COSH*100+CPHVB_BOOL:
                    iter<cphvb_bool>( instr, &score_cosh );
                    break;
                case CPHVB_COSH*100+CPHVB_INT8:
                    iter<cphvb_int8>( instr, &score_cosh );
                    break;
                case CPHVB_COSH*100+CPHVB_INT16:
                    iter<cphvb_int16>( instr, &score_cosh );
                    break;
                case CPHVB_COSH*100+CPHVB_INT32:
                    iter<cphvb_int32>( instr, &score_cosh );
                    break;
                case CPHVB_COSH*100+CPHVB_INT64:
                    iter<cphvb_int64>( instr, &score_cosh );
                    break;
                case CPHVB_COSH*100+CPHVB_UINT8:
                    iter<cphvb_uint8>( instr, &score_cosh );
                    break;
                case CPHVB_COSH*100+CPHVB_UINT16:
                    iter<cphvb_uint16>( instr, &score_cosh );
                    break;
                case CPHVB_COSH*100+CPHVB_UINT32:
                    iter<cphvb_uint32>( instr, &score_cosh );
                    break;
                case CPHVB_COSH*100+CPHVB_UINT64:
                    iter<cphvb_uint64>( instr, &score_cosh );
                    break;
                case CPHVB_COSH*100+CPHVB_FLOAT32:
                    iter<cphvb_float32>( instr, &score_cosh );
                    break;
                case CPHVB_COSH*100+CPHVB_FLOAT64:
                    iter<cphvb_float64>( instr, &score_cosh );
                    break;
                case CPHVB_TANH*100+CPHVB_BOOL:
                    iter<cphvb_bool>( instr, &score_tanh );
                    break;
                case CPHVB_TANH*100+CPHVB_INT8:
                    iter<cphvb_int8>( instr, &score_tanh );
                    break;
                case CPHVB_TANH*100+CPHVB_INT16:
                    iter<cphvb_int16>( instr, &score_tanh );
                    break;
                case CPHVB_TANH*100+CPHVB_INT32:
                    iter<cphvb_int32>( instr, &score_tanh );
                    break;
                case CPHVB_TANH*100+CPHVB_INT64:
                    iter<cphvb_int64>( instr, &score_tanh );
                    break;
                case CPHVB_TANH*100+CPHVB_UINT8:
                    iter<cphvb_uint8>( instr, &score_tanh );
                    break;
                case CPHVB_TANH*100+CPHVB_UINT16:
                    iter<cphvb_uint16>( instr, &score_tanh );
                    break;
                case CPHVB_TANH*100+CPHVB_UINT32:
                    iter<cphvb_uint32>( instr, &score_tanh );
                    break;
                case CPHVB_TANH*100+CPHVB_UINT64:
                    iter<cphvb_uint64>( instr, &score_tanh );
                    break;
                case CPHVB_TANH*100+CPHVB_FLOAT32:
                    iter<cphvb_float32>( instr, &score_tanh );
                    break;
                case CPHVB_TANH*100+CPHVB_FLOAT64:
                    iter<cphvb_float64>( instr, &score_tanh );
                    break;
                case CPHVB_ARCSINH*100+CPHVB_BOOL:
                    iter<cphvb_bool>( instr, &score_arcsinh );
                    break;
                case CPHVB_ARCSINH*100+CPHVB_INT8:
                    iter<cphvb_int8>( instr, &score_arcsinh );
                    break;
                case CPHVB_ARCSINH*100+CPHVB_INT16:
                    iter<cphvb_int16>( instr, &score_arcsinh );
                    break;
                case CPHVB_ARCSINH*100+CPHVB_INT32:
                    iter<cphvb_int32>( instr, &score_arcsinh );
                    break;
                case CPHVB_ARCSINH*100+CPHVB_INT64:
                    iter<cphvb_int64>( instr, &score_arcsinh );
                    break;
                case CPHVB_ARCSINH*100+CPHVB_UINT8:
                    iter<cphvb_uint8>( instr, &score_arcsinh );
                    break;
                case CPHVB_ARCSINH*100+CPHVB_UINT16:
                    iter<cphvb_uint16>( instr, &score_arcsinh );
                    break;
                case CPHVB_ARCSINH*100+CPHVB_UINT32:
                    iter<cphvb_uint32>( instr, &score_arcsinh );
                    break;
                case CPHVB_ARCSINH*100+CPHVB_UINT64:
                    iter<cphvb_uint64>( instr, &score_arcsinh );
                    break;
                case CPHVB_ARCSINH*100+CPHVB_FLOAT32:
                    iter<cphvb_float32>( instr, &score_arcsinh );
                    break;
                case CPHVB_ARCSINH*100+CPHVB_FLOAT64:
                    iter<cphvb_float64>( instr, &score_arcsinh );
                    break;
                case CPHVB_ARCCOSH*100+CPHVB_BOOL:
                    iter<cphvb_bool>( instr, &score_arccosh );
                    break;
                case CPHVB_ARCCOSH*100+CPHVB_INT8:
                    iter<cphvb_int8>( instr, &score_arccosh );
                    break;
                case CPHVB_ARCCOSH*100+CPHVB_INT16:
                    iter<cphvb_int16>( instr, &score_arccosh );
                    break;
                case CPHVB_ARCCOSH*100+CPHVB_INT32:
                    iter<cphvb_int32>( instr, &score_arccosh );
                    break;
                case CPHVB_ARCCOSH*100+CPHVB_INT64:
                    iter<cphvb_int64>( instr, &score_arccosh );
                    break;
                case CPHVB_ARCCOSH*100+CPHVB_UINT8:
                    iter<cphvb_uint8>( instr, &score_arccosh );
                    break;
                case CPHVB_ARCCOSH*100+CPHVB_UINT16:
                    iter<cphvb_uint16>( instr, &score_arccosh );
                    break;
                case CPHVB_ARCCOSH*100+CPHVB_UINT32:
                    iter<cphvb_uint32>( instr, &score_arccosh );
                    break;
                case CPHVB_ARCCOSH*100+CPHVB_UINT64:
                    iter<cphvb_uint64>( instr, &score_arccosh );
                    break;
                case CPHVB_ARCCOSH*100+CPHVB_FLOAT32:
                    iter<cphvb_float32>( instr, &score_arccosh );
                    break;
                case CPHVB_ARCCOSH*100+CPHVB_FLOAT64:
                    iter<cphvb_float64>( instr, &score_arccosh );
                    break;
                case CPHVB_ARCTANH*100+CPHVB_BOOL:
                    iter<cphvb_bool>( instr, &score_arctanh );
                    break;
                case CPHVB_ARCTANH*100+CPHVB_INT8:
                    iter<cphvb_int8>( instr, &score_arctanh );
                    break;
                case CPHVB_ARCTANH*100+CPHVB_INT16:
                    iter<cphvb_int16>( instr, &score_arctanh );
                    break;
                case CPHVB_ARCTANH*100+CPHVB_INT32:
                    iter<cphvb_int32>( instr, &score_arctanh );
                    break;
                case CPHVB_ARCTANH*100+CPHVB_INT64:
                    iter<cphvb_int64>( instr, &score_arctanh );
                    break;
                case CPHVB_ARCTANH*100+CPHVB_UINT8:
                    iter<cphvb_uint8>( instr, &score_arctanh );
                    break;
                case CPHVB_ARCTANH*100+CPHVB_UINT16:
                    iter<cphvb_uint16>( instr, &score_arctanh );
                    break;
                case CPHVB_ARCTANH*100+CPHVB_UINT32:
                    iter<cphvb_uint32>( instr, &score_arctanh );
                    break;
                case CPHVB_ARCTANH*100+CPHVB_UINT64:
                    iter<cphvb_uint64>( instr, &score_arctanh );
                    break;
                case CPHVB_ARCTANH*100+CPHVB_FLOAT32:
                    iter<cphvb_float32>( instr, &score_arctanh );
                    break;
                case CPHVB_ARCTANH*100+CPHVB_FLOAT64:
                    iter<cphvb_float64>( instr, &score_arctanh );
                    break;
                case CPHVB_DEG2RAD*100+CPHVB_BOOL:
                    iter<cphvb_bool>( instr, &score_deg2rad );
                    break;
                case CPHVB_DEG2RAD*100+CPHVB_INT8:
                    iter<cphvb_int8>( instr, &score_deg2rad );
                    break;
                case CPHVB_DEG2RAD*100+CPHVB_INT16:
                    iter<cphvb_int16>( instr, &score_deg2rad );
                    break;
                case CPHVB_DEG2RAD*100+CPHVB_INT32:
                    iter<cphvb_int32>( instr, &score_deg2rad );
                    break;
                case CPHVB_DEG2RAD*100+CPHVB_INT64:
                    iter<cphvb_int64>( instr, &score_deg2rad );
                    break;
                case CPHVB_DEG2RAD*100+CPHVB_UINT8:
                    iter<cphvb_uint8>( instr, &score_deg2rad );
                    break;
                case CPHVB_DEG2RAD*100+CPHVB_UINT16:
                    iter<cphvb_uint16>( instr, &score_deg2rad );
                    break;
                case CPHVB_DEG2RAD*100+CPHVB_UINT32:
                    iter<cphvb_uint32>( instr, &score_deg2rad );
                    break;
                case CPHVB_DEG2RAD*100+CPHVB_UINT64:
                    iter<cphvb_uint64>( instr, &score_deg2rad );
                    break;
                case CPHVB_DEG2RAD*100+CPHVB_FLOAT32:
                    iter<cphvb_float32>( instr, &score_deg2rad );
                    break;
                case CPHVB_DEG2RAD*100+CPHVB_FLOAT64:
                    iter<cphvb_float64>( instr, &score_deg2rad );
                    break;
                case CPHVB_RAD2DEG*100+CPHVB_BOOL:
                    iter<cphvb_bool>( instr, &score_rad2deg );
                    break;
                case CPHVB_RAD2DEG*100+CPHVB_INT8:
                    iter<cphvb_int8>( instr, &score_rad2deg );
                    break;
                case CPHVB_RAD2DEG*100+CPHVB_INT16:
                    iter<cphvb_int16>( instr, &score_rad2deg );
                    break;
                case CPHVB_RAD2DEG*100+CPHVB_INT32:
                    iter<cphvb_int32>( instr, &score_rad2deg );
                    break;
                case CPHVB_RAD2DEG*100+CPHVB_INT64:
                    iter<cphvb_int64>( instr, &score_rad2deg );
                    break;
                case CPHVB_RAD2DEG*100+CPHVB_UINT8:
                    iter<cphvb_uint8>( instr, &score_rad2deg );
                    break;
                case CPHVB_RAD2DEG*100+CPHVB_UINT16:
                    iter<cphvb_uint16>( instr, &score_rad2deg );
                    break;
                case CPHVB_RAD2DEG*100+CPHVB_UINT32:
                    iter<cphvb_uint32>( instr, &score_rad2deg );
                    break;
                case CPHVB_RAD2DEG*100+CPHVB_UINT64:
                    iter<cphvb_uint64>( instr, &score_rad2deg );
                    break;
                case CPHVB_RAD2DEG*100+CPHVB_FLOAT32:
                    iter<cphvb_float32>( instr, &score_rad2deg );
                    break;
                case CPHVB_RAD2DEG*100+CPHVB_FLOAT64:
                    iter<cphvb_float64>( instr, &score_rad2deg );
                    break;
                case CPHVB_LOGICAL_NOT*100+CPHVB_BOOL:
                    iter<cphvb_bool>( instr, &score_logical_not );
                    break;
                case CPHVB_LOGICAL_NOT*100+CPHVB_INT8:
                    iter<cphvb_int8>( instr, &score_logical_not );
                    break;
                case CPHVB_LOGICAL_NOT*100+CPHVB_INT16:
                    iter<cphvb_int16>( instr, &score_logical_not );
                    break;
                case CPHVB_LOGICAL_NOT*100+CPHVB_INT32:
                    iter<cphvb_int32>( instr, &score_logical_not );
                    break;
                case CPHVB_LOGICAL_NOT*100+CPHVB_INT64:
                    iter<cphvb_int64>( instr, &score_logical_not );
                    break;
                case CPHVB_LOGICAL_NOT*100+CPHVB_UINT8:
                    iter<cphvb_uint8>( instr, &score_logical_not );
                    break;
                case CPHVB_LOGICAL_NOT*100+CPHVB_UINT16:
                    iter<cphvb_uint16>( instr, &score_logical_not );
                    break;
                case CPHVB_LOGICAL_NOT*100+CPHVB_UINT32:
                    iter<cphvb_uint32>( instr, &score_logical_not );
                    break;
                case CPHVB_LOGICAL_NOT*100+CPHVB_UINT64:
                    iter<cphvb_uint64>( instr, &score_logical_not );
                    break;
                case CPHVB_LOGICAL_NOT*100+CPHVB_FLOAT32:
                    iter<cphvb_float32>( instr, &score_logical_not );
                    break;
                case CPHVB_LOGICAL_NOT*100+CPHVB_FLOAT64:
                    iter<cphvb_float64>( instr, &score_logical_not );
                    break;
                case CPHVB_INVERT*100+CPHVB_BOOL:
                    iter<cphvb_bool>( instr, &score_invert );
                    break;
                case CPHVB_INVERT*100+CPHVB_INT8:
                    iter<cphvb_int8>( instr, &score_invert );
                    break;
                case CPHVB_INVERT*100+CPHVB_INT16:
                    iter<cphvb_int16>( instr, &score_invert );
                    break;
                case CPHVB_INVERT*100+CPHVB_INT32:
                    iter<cphvb_int32>( instr, &score_invert );
                    break;
                case CPHVB_INVERT*100+CPHVB_INT64:
                    iter<cphvb_int64>( instr, &score_invert );
                    break;
                case CPHVB_INVERT*100+CPHVB_UINT8:
                    iter<cphvb_uint8>( instr, &score_invert );
                    break;
                case CPHVB_INVERT*100+CPHVB_UINT16:
                    iter<cphvb_uint16>( instr, &score_invert );
                    break;
                case CPHVB_INVERT*100+CPHVB_UINT32:
                    iter<cphvb_uint32>( instr, &score_invert );
                    break;
                case CPHVB_INVERT*100+CPHVB_UINT64:
                    iter<cphvb_uint64>( instr, &score_invert );
                    break;
                case CPHVB_INVERT*100+CPHVB_FLOAT32:
                    iter<cphvb_float32>( instr, &score_invert );
                    break;
                case CPHVB_INVERT*100+CPHVB_FLOAT64:
                    iter<cphvb_float64>( instr, &score_invert );
                    break;
                case CPHVB_ISFINITE*100+CPHVB_BOOL:
                    iter<cphvb_bool>( instr, &score_isfinite );
                    break;
                case CPHVB_ISFINITE*100+CPHVB_INT8:
                    iter<cphvb_int8>( instr, &score_isfinite );
                    break;
                case CPHVB_ISFINITE*100+CPHVB_INT16:
                    iter<cphvb_int16>( instr, &score_isfinite );
                    break;
                case CPHVB_ISFINITE*100+CPHVB_INT32:
                    iter<cphvb_int32>( instr, &score_isfinite );
                    break;
                case CPHVB_ISFINITE*100+CPHVB_INT64:
                    iter<cphvb_int64>( instr, &score_isfinite );
                    break;
                case CPHVB_ISFINITE*100+CPHVB_UINT8:
                    iter<cphvb_uint8>( instr, &score_isfinite );
                    break;
                case CPHVB_ISFINITE*100+CPHVB_UINT16:
                    iter<cphvb_uint16>( instr, &score_isfinite );
                    break;
                case CPHVB_ISFINITE*100+CPHVB_UINT32:
                    iter<cphvb_uint32>( instr, &score_isfinite );
                    break;
                case CPHVB_ISFINITE*100+CPHVB_UINT64:
                    iter<cphvb_uint64>( instr, &score_isfinite );
                    break;
                case CPHVB_ISFINITE*100+CPHVB_FLOAT32:
                    iter<cphvb_float32>( instr, &score_isfinite );
                    break;
                case CPHVB_ISFINITE*100+CPHVB_FLOAT64:
                    iter<cphvb_float64>( instr, &score_isfinite );
                    break;
                case CPHVB_ISINF*100+CPHVB_BOOL:
                    iter<cphvb_bool>( instr, &score_isinf );
                    break;
                case CPHVB_ISINF*100+CPHVB_INT8:
                    iter<cphvb_int8>( instr, &score_isinf );
                    break;
                case CPHVB_ISINF*100+CPHVB_INT16:
                    iter<cphvb_int16>( instr, &score_isinf );
                    break;
                case CPHVB_ISINF*100+CPHVB_INT32:
                    iter<cphvb_int32>( instr, &score_isinf );
                    break;
                case CPHVB_ISINF*100+CPHVB_INT64:
                    iter<cphvb_int64>( instr, &score_isinf );
                    break;
                case CPHVB_ISINF*100+CPHVB_UINT8:
                    iter<cphvb_uint8>( instr, &score_isinf );
                    break;
                case CPHVB_ISINF*100+CPHVB_UINT16:
                    iter<cphvb_uint16>( instr, &score_isinf );
                    break;
                case CPHVB_ISINF*100+CPHVB_UINT32:
                    iter<cphvb_uint32>( instr, &score_isinf );
                    break;
                case CPHVB_ISINF*100+CPHVB_UINT64:
                    iter<cphvb_uint64>( instr, &score_isinf );
                    break;
                case CPHVB_ISINF*100+CPHVB_FLOAT32:
                    iter<cphvb_float32>( instr, &score_isinf );
                    break;
                case CPHVB_ISINF*100+CPHVB_FLOAT64:
                    iter<cphvb_float64>( instr, &score_isinf );
                    break;
                case CPHVB_ISNAN*100+CPHVB_BOOL:
                    iter<cphvb_bool>( instr, &score_isnan );
                    break;
                case CPHVB_ISNAN*100+CPHVB_INT8:
                    iter<cphvb_int8>( instr, &score_isnan );
                    break;
                case CPHVB_ISNAN*100+CPHVB_INT16:
                    iter<cphvb_int16>( instr, &score_isnan );
                    break;
                case CPHVB_ISNAN*100+CPHVB_INT32:
                    iter<cphvb_int32>( instr, &score_isnan );
                    break;
                case CPHVB_ISNAN*100+CPHVB_INT64:
                    iter<cphvb_int64>( instr, &score_isnan );
                    break;
                case CPHVB_ISNAN*100+CPHVB_UINT8:
                    iter<cphvb_uint8>( instr, &score_isnan );
                    break;
                case CPHVB_ISNAN*100+CPHVB_UINT16:
                    iter<cphvb_uint16>( instr, &score_isnan );
                    break;
                case CPHVB_ISNAN*100+CPHVB_UINT32:
                    iter<cphvb_uint32>( instr, &score_isnan );
                    break;
                case CPHVB_ISNAN*100+CPHVB_UINT64:
                    iter<cphvb_uint64>( instr, &score_isnan );
                    break;
                case CPHVB_ISNAN*100+CPHVB_FLOAT32:
                    iter<cphvb_float32>( instr, &score_isnan );
                    break;
                case CPHVB_ISNAN*100+CPHVB_FLOAT64:
                    iter<cphvb_float64>( instr, &score_isnan );
                    break;
                case CPHVB_SIGNBIT*100+CPHVB_BOOL:
                    iter<cphvb_bool>( instr, &score_signbit );
                    break;
                case CPHVB_SIGNBIT*100+CPHVB_INT8:
                    iter<cphvb_int8>( instr, &score_signbit );
                    break;
                case CPHVB_SIGNBIT*100+CPHVB_INT16:
                    iter<cphvb_int16>( instr, &score_signbit );
                    break;
                case CPHVB_SIGNBIT*100+CPHVB_INT32:
                    iter<cphvb_int32>( instr, &score_signbit );
                    break;
                case CPHVB_SIGNBIT*100+CPHVB_INT64:
                    iter<cphvb_int64>( instr, &score_signbit );
                    break;
                case CPHVB_SIGNBIT*100+CPHVB_UINT8:
                    iter<cphvb_uint8>( instr, &score_signbit );
                    break;
                case CPHVB_SIGNBIT*100+CPHVB_UINT16:
                    iter<cphvb_uint16>( instr, &score_signbit );
                    break;
                case CPHVB_SIGNBIT*100+CPHVB_UINT32:
                    iter<cphvb_uint32>( instr, &score_signbit );
                    break;
                case CPHVB_SIGNBIT*100+CPHVB_UINT64:
                    iter<cphvb_uint64>( instr, &score_signbit );
                    break;
                case CPHVB_SIGNBIT*100+CPHVB_FLOAT32:
                    iter<cphvb_float32>( instr, &score_signbit );
                    break;
                case CPHVB_SIGNBIT*100+CPHVB_FLOAT64:
                    iter<cphvb_float64>( instr, &score_signbit );
                    break;
                case CPHVB_FLOOR*100+CPHVB_BOOL:
                    iter<cphvb_bool>( instr, &score_floor );
                    break;
                case CPHVB_FLOOR*100+CPHVB_INT8:
                    iter<cphvb_int8>( instr, &score_floor );
                    break;
                case CPHVB_FLOOR*100+CPHVB_INT16:
                    iter<cphvb_int16>( instr, &score_floor );
                    break;
                case CPHVB_FLOOR*100+CPHVB_INT32:
                    iter<cphvb_int32>( instr, &score_floor );
                    break;
                case CPHVB_FLOOR*100+CPHVB_INT64:
                    iter<cphvb_int64>( instr, &score_floor );
                    break;
                case CPHVB_FLOOR*100+CPHVB_UINT8:
                    iter<cphvb_uint8>( instr, &score_floor );
                    break;
                case CPHVB_FLOOR*100+CPHVB_UINT16:
                    iter<cphvb_uint16>( instr, &score_floor );
                    break;
                case CPHVB_FLOOR*100+CPHVB_UINT32:
                    iter<cphvb_uint32>( instr, &score_floor );
                    break;
                case CPHVB_FLOOR*100+CPHVB_UINT64:
                    iter<cphvb_uint64>( instr, &score_floor );
                    break;
                case CPHVB_FLOOR*100+CPHVB_FLOAT32:
                    iter<cphvb_float32>( instr, &score_floor );
                    break;
                case CPHVB_FLOOR*100+CPHVB_FLOAT64:
                    iter<cphvb_float64>( instr, &score_floor );
                    break;
                case CPHVB_CEIL*100+CPHVB_BOOL:
                    iter<cphvb_bool>( instr, &score_ceil );
                    break;
                case CPHVB_CEIL*100+CPHVB_INT8:
                    iter<cphvb_int8>( instr, &score_ceil );
                    break;
                case CPHVB_CEIL*100+CPHVB_INT16:
                    iter<cphvb_int16>( instr, &score_ceil );
                    break;
                case CPHVB_CEIL*100+CPHVB_INT32:
                    iter<cphvb_int32>( instr, &score_ceil );
                    break;
                case CPHVB_CEIL*100+CPHVB_INT64:
                    iter<cphvb_int64>( instr, &score_ceil );
                    break;
                case CPHVB_CEIL*100+CPHVB_UINT8:
                    iter<cphvb_uint8>( instr, &score_ceil );
                    break;
                case CPHVB_CEIL*100+CPHVB_UINT16:
                    iter<cphvb_uint16>( instr, &score_ceil );
                    break;
                case CPHVB_CEIL*100+CPHVB_UINT32:
                    iter<cphvb_uint32>( instr, &score_ceil );
                    break;
                case CPHVB_CEIL*100+CPHVB_UINT64:
                    iter<cphvb_uint64>( instr, &score_ceil );
                    break;
                case CPHVB_CEIL*100+CPHVB_FLOAT32:
                    iter<cphvb_float32>( instr, &score_ceil );
                    break;
                case CPHVB_CEIL*100+CPHVB_FLOAT64:
                    iter<cphvb_float64>( instr, &score_ceil );
                    break;
                case CPHVB_TRUNC*100+CPHVB_BOOL:
                    iter<cphvb_bool>( instr, &score_trunc );
                    break;
                case CPHVB_TRUNC*100+CPHVB_INT8:
                    iter<cphvb_int8>( instr, &score_trunc );
                    break;
                case CPHVB_TRUNC*100+CPHVB_INT16:
                    iter<cphvb_int16>( instr, &score_trunc );
                    break;
                case CPHVB_TRUNC*100+CPHVB_INT32:
                    iter<cphvb_int32>( instr, &score_trunc );
                    break;
                case CPHVB_TRUNC*100+CPHVB_INT64:
                    iter<cphvb_int64>( instr, &score_trunc );
                    break;
                case CPHVB_TRUNC*100+CPHVB_UINT8:
                    iter<cphvb_uint8>( instr, &score_trunc );
                    break;
                case CPHVB_TRUNC*100+CPHVB_UINT16:
                    iter<cphvb_uint16>( instr, &score_trunc );
                    break;
                case CPHVB_TRUNC*100+CPHVB_UINT32:
                    iter<cphvb_uint32>( instr, &score_trunc );
                    break;
                case CPHVB_TRUNC*100+CPHVB_UINT64:
                    iter<cphvb_uint64>( instr, &score_trunc );
                    break;
                case CPHVB_TRUNC*100+CPHVB_FLOAT32:
                    iter<cphvb_float32>( instr, &score_trunc );
                    break;
                case CPHVB_TRUNC*100+CPHVB_FLOAT64:
                    iter<cphvb_float64>( instr, &score_trunc );
                    break;
                case CPHVB_ISREAL*100+CPHVB_BOOL:
                    iter<cphvb_bool>( instr, &score_isreal );
                    break;
                case CPHVB_ISREAL*100+CPHVB_INT8:
                    iter<cphvb_int8>( instr, &score_isreal );
                    break;
                case CPHVB_ISREAL*100+CPHVB_INT16:
                    iter<cphvb_int16>( instr, &score_isreal );
                    break;
                case CPHVB_ISREAL*100+CPHVB_INT32:
                    iter<cphvb_int32>( instr, &score_isreal );
                    break;
                case CPHVB_ISREAL*100+CPHVB_INT64:
                    iter<cphvb_int64>( instr, &score_isreal );
                    break;
                case CPHVB_ISREAL*100+CPHVB_UINT8:
                    iter<cphvb_uint8>( instr, &score_isreal );
                    break;
                case CPHVB_ISREAL*100+CPHVB_UINT16:
                    iter<cphvb_uint16>( instr, &score_isreal );
                    break;
                case CPHVB_ISREAL*100+CPHVB_UINT32:
                    iter<cphvb_uint32>( instr, &score_isreal );
                    break;
                case CPHVB_ISREAL*100+CPHVB_UINT64:
                    iter<cphvb_uint64>( instr, &score_isreal );
                    break;
                case CPHVB_ISREAL*100+CPHVB_FLOAT32:
                    iter<cphvb_float32>( instr, &score_isreal );
                    break;
                case CPHVB_ISREAL*100+CPHVB_FLOAT64:
                    iter<cphvb_float64>( instr, &score_isreal );
                    break;
                case CPHVB_ISCOMPLEX*100+CPHVB_BOOL:
                    iter<cphvb_bool>( instr, &score_iscomplex );
                    break;
                case CPHVB_ISCOMPLEX*100+CPHVB_INT8:
                    iter<cphvb_int8>( instr, &score_iscomplex );
                    break;
                case CPHVB_ISCOMPLEX*100+CPHVB_INT16:
                    iter<cphvb_int16>( instr, &score_iscomplex );
                    break;
                case CPHVB_ISCOMPLEX*100+CPHVB_INT32:
                    iter<cphvb_int32>( instr, &score_iscomplex );
                    break;
                case CPHVB_ISCOMPLEX*100+CPHVB_INT64:
                    iter<cphvb_int64>( instr, &score_iscomplex );
                    break;
                case CPHVB_ISCOMPLEX*100+CPHVB_UINT8:
                    iter<cphvb_uint8>( instr, &score_iscomplex );
                    break;
                case CPHVB_ISCOMPLEX*100+CPHVB_UINT16:
                    iter<cphvb_uint16>( instr, &score_iscomplex );
                    break;
                case CPHVB_ISCOMPLEX*100+CPHVB_UINT32:
                    iter<cphvb_uint32>( instr, &score_iscomplex );
                    break;
                case CPHVB_ISCOMPLEX*100+CPHVB_UINT64:
                    iter<cphvb_uint64>( instr, &score_iscomplex );
                    break;
                case CPHVB_ISCOMPLEX*100+CPHVB_FLOAT32:
                    iter<cphvb_float32>( instr, &score_iscomplex );
                    break;
                case CPHVB_ISCOMPLEX*100+CPHVB_FLOAT64:
                    iter<cphvb_float64>( instr, &score_iscomplex );
                    break;
                case CPHVB_MODF*100+CPHVB_BOOL:
                    iter<cphvb_bool>( instr, &score_modf );
                    break;
                case CPHVB_MODF*100+CPHVB_INT8:
                    iter<cphvb_int8>( instr, &score_modf );
                    break;
                case CPHVB_MODF*100+CPHVB_INT16:
                    iter<cphvb_int16>( instr, &score_modf );
                    break;
                case CPHVB_MODF*100+CPHVB_INT32:
                    iter<cphvb_int32>( instr, &score_modf );
                    break;
                case CPHVB_MODF*100+CPHVB_INT64:
                    iter<cphvb_int64>( instr, &score_modf );
                    break;
                case CPHVB_MODF*100+CPHVB_UINT8:
                    iter<cphvb_uint8>( instr, &score_modf );
                    break;
                case CPHVB_MODF*100+CPHVB_UINT16:
                    iter<cphvb_uint16>( instr, &score_modf );
                    break;
                case CPHVB_MODF*100+CPHVB_UINT32:
                    iter<cphvb_uint32>( instr, &score_modf );
                    break;
                case CPHVB_MODF*100+CPHVB_UINT64:
                    iter<cphvb_uint64>( instr, &score_modf );
                    break;
                case CPHVB_MODF*100+CPHVB_FLOAT32:
                    iter<cphvb_float32>( instr, &score_modf );
                    break;
                case CPHVB_MODF*100+CPHVB_FLOAT64:
                    iter<cphvb_float64>( instr, &score_modf );
                    break;
                case CPHVB_FREXP*100+CPHVB_BOOL:
                    iter<cphvb_bool>( instr, &score_frexp );
                    break;
                case CPHVB_FREXP*100+CPHVB_INT8:
                    iter<cphvb_int8>( instr, &score_frexp );
                    break;
                case CPHVB_FREXP*100+CPHVB_INT16:
                    iter<cphvb_int16>( instr, &score_frexp );
                    break;
                case CPHVB_FREXP*100+CPHVB_INT32:
                    iter<cphvb_int32>( instr, &score_frexp );
                    break;
                case CPHVB_FREXP*100+CPHVB_INT64:
                    iter<cphvb_int64>( instr, &score_frexp );
                    break;
                case CPHVB_FREXP*100+CPHVB_UINT8:
                    iter<cphvb_uint8>( instr, &score_frexp );
                    break;
                case CPHVB_FREXP*100+CPHVB_UINT16:
                    iter<cphvb_uint16>( instr, &score_frexp );
                    break;
                case CPHVB_FREXP*100+CPHVB_UINT32:
                    iter<cphvb_uint32>( instr, &score_frexp );
                    break;
                case CPHVB_FREXP*100+CPHVB_UINT64:
                    iter<cphvb_uint64>( instr, &score_frexp );
                    break;
                case CPHVB_FREXP*100+CPHVB_FLOAT32:
                    iter<cphvb_float32>( instr, &score_frexp );
                    break;
                case CPHVB_FREXP*100+CPHVB_FLOAT64:
                    iter<cphvb_float64>( instr, &score_frexp );
                    break;
                case CPHVB_RANDOM*100+CPHVB_BOOL:
                    iter<cphvb_bool>( instr, &score_random );
                    break;
                case CPHVB_RANDOM*100+CPHVB_INT8:
                    iter<cphvb_int8>( instr, &score_random );
                    break;
                case CPHVB_RANDOM*100+CPHVB_INT16:
                    iter<cphvb_int16>( instr, &score_random );
                    break;
                case CPHVB_RANDOM*100+CPHVB_INT32:
                    iter<cphvb_int32>( instr, &score_random );
                    break;
                case CPHVB_RANDOM*100+CPHVB_INT64:
                    iter<cphvb_int64>( instr, &score_random );
                    break;
                case CPHVB_RANDOM*100+CPHVB_UINT8:
                    iter<cphvb_uint8>( instr, &score_random );
                    break;
                case CPHVB_RANDOM*100+CPHVB_UINT16:
                    iter<cphvb_uint16>( instr, &score_random );
                    break;
                case CPHVB_RANDOM*100+CPHVB_UINT32:
                    iter<cphvb_uint32>( instr, &score_random );
                    break;
                case CPHVB_RANDOM*100+CPHVB_UINT64:
                    iter<cphvb_uint64>( instr, &score_random );
                    break;
                case CPHVB_RANDOM*100+CPHVB_FLOAT32:
                    iter<cphvb_float32>( instr, &score_random );
                    break;
                case CPHVB_RANDOM*100+CPHVB_FLOAT64:
                    iter<cphvb_float64>( instr, &score_random );
                    break;
                case CPHVB_ARANGE*100+CPHVB_BOOL:
                    iter<cphvb_bool>( instr, &score_arange );
                    break;
                case CPHVB_ARANGE*100+CPHVB_INT8:
                    iter<cphvb_int8>( instr, &score_arange );
                    break;
                case CPHVB_ARANGE*100+CPHVB_INT16:
                    iter<cphvb_int16>( instr, &score_arange );
                    break;
                case CPHVB_ARANGE*100+CPHVB_INT32:
                    iter<cphvb_int32>( instr, &score_arange );
                    break;
                case CPHVB_ARANGE*100+CPHVB_INT64:
                    iter<cphvb_int64>( instr, &score_arange );
                    break;
                case CPHVB_ARANGE*100+CPHVB_UINT8:
                    iter<cphvb_uint8>( instr, &score_arange );
                    break;
                case CPHVB_ARANGE*100+CPHVB_UINT16:
                    iter<cphvb_uint16>( instr, &score_arange );
                    break;
                case CPHVB_ARANGE*100+CPHVB_UINT32:
                    iter<cphvb_uint32>( instr, &score_arange );
                    break;
                case CPHVB_ARANGE*100+CPHVB_UINT64:
                    iter<cphvb_uint64>( instr, &score_arange );
                    break;
                case CPHVB_ARANGE*100+CPHVB_FLOAT32:
                    iter<cphvb_float32>( instr, &score_arange );
                    break;
                case CPHVB_ARANGE*100+CPHVB_FLOAT64:
                    iter<cphvb_float64>( instr, &score_arange );
                    break;

                default:                // Unsupported instruction
                    fprintf(
                        stderr, 
                        "cphvb_ve_score_execute() encountered an unknown opcode: %s \
                        in combination with argument types.",
                        cphvb_opcode_text( instr->opcode )
                    );
                    res = CPHVB_INST_NOT_SUPPORTED;

            }

    }

    return res;

}

