#include <sstream>
#include <string>
#include "codegen.hpp"
//
//  NOTE: This file is autogenerated based on the tac-definition.
//        You should therefore not edit it manually.
//
using namespace std;
using namespace kp::core;

namespace kp{
namespace engine{
namespace codegen{

string Walker::oper_neutral_element(KP_OPERATOR oper, KP_ETYPE etype)
{
    switch(oper) {
        case KP_ADD:               return "0";
        case KP_MULTIPLY:          return "1";
        case KP_MAXIMUM:
            switch(etype) {
                case KP_BOOL:      return "0";
                case KP_INT8:      return "INT8_MIN";
                case KP_INT16:     return "INT16_MIN";
                case KP_INT32:     return "INT32_MIN";
                case KP_INT64:     return "INT64_MIN";
                case KP_UINT8:     return "UINT8_MIN";
                case KP_UINT16:    return "UINT16_MIN";
                case KP_UINT32:    return "UINT32_MIN";
                case KP_UINT64:    return "UINT64_MIN";
                case KP_FLOAT32:   return "FLT_MIN";
                case KP_FLOAT64:   return "DBL_MIN";
                default:        return "UNKNOWN_NEUTRAL_FOR_MAXIMUM_OF_GIVEN_TYPE";
            }
        case KP_MINIMUM:
            switch(etype) {
                case KP_BOOL:      return "1";
                case KP_INT8:      return "INT8_MAX";
                case KP_INT16:     return "INT16_MAX";
                case KP_INT32:     return "INT32_MAX";
                case KP_INT64:     return "INT64_MAX";
                case KP_UINT8:     return "UINT8_MAX";
                case KP_UINT16:    return "UINT16_MAX";
                case KP_UINT32:    return "UINT32_MAX";
                case KP_UINT64:    return "UINT64_MAX";
                case KP_FLOAT32:   return "FLT_MAX";
                case KP_FLOAT64:   return "DBL_MAX";
                default:        return "UNKNOWN_NEUTRAL_FOR_MINIMUM_OF_GIVEN_TYPE";
            }
        case KP_LOGICAL_AND:       return "1";
        case KP_LOGICAL_OR:        return "0";
        case KP_LOGICAL_XOR:       return "0";
        case KP_BITWISE_AND:
            switch(etype) {
                case KP_BOOL:      return "1";
                case KP_INT8:      return "-1";
                case KP_INT16:     return "-1";
                case KP_INT32:     return "-1";
                case KP_INT64:     return "-1";
                case KP_UINT8:     return "UINT8_MAX";
                case KP_UINT16:    return "UINT16_MAX";
                case KP_UINT32:    return "UINT32_MAX";
                case KP_UINT64:    return "UINT64_MAX";
                default:        return "UNKNOWN_NEUTRAL_FOR_BITWISE_AND_OF_GIVEN_TYPE";
            }
        case KP_BITWISE_OR:        return "0";
        case KP_BITWISE_XOR:       return "0";
        default:                return "UNKNOWN_NEUTRAL_FOR_OPERATOR";
    }
}

string Walker::oper_description(kp_tac tac)
{
    stringstream ss;
    ss << operator_text(tac.oper) << " (";
    switch(core::tac_noperands(tac)) {
        case 3:
            ss << layout_text(kernel_.operand_glb(tac.out).meta().layout);
            ss << ", ";
            ss << layout_text(kernel_.operand_glb(tac.in1).meta().layout);
            ss << ", ";
            ss << layout_text(kernel_.operand_glb(tac.in2).meta().layout);
            break;
        case 2:
            ss << layout_text(kernel_.operand_glb(tac.out).meta().layout);
            ss << ", ";
            ss << layout_text(kernel_.operand_glb(tac.in1).meta().layout);
            break;
        case 1:
            ss << layout_text(kernel_.operand_glb(tac.out).meta().layout);
            break;
        default:
            break;
    }
    ss << ")";
    return ss.str();
}

string Walker::oper(KP_OPERATOR oper, KP_ETYPE etype, string in1, string in2)
{
    switch(oper) {
        case KP_ABSOLUTE:
            switch(etype) {
                case KP_COMPLEX128:    return _cabs(in1);
                case KP_COMPLEX64:     return _cabsf(in1);
                default:            return _abs(in1);
            }
        case KP_ADD:                   return _add(in1, in2);
        case KP_ARCCOS:
            switch(etype) {
                case KP_COMPLEX128:    return _cacos(in1);
                case KP_COMPLEX64:     return _cacosf(in1);
                default:            return _acos(in1);
            }
        case KP_ARCCOSH:
            switch(etype) {
                case KP_COMPLEX128:    return _cacosh(in1);
                case KP_COMPLEX64:     return _cacosf(in1);
                default:            return _acosh(in1);
            }
        case KP_ARCSIN:
            switch(etype) {
                case KP_COMPLEX128:    return _casin(in1);
                case KP_COMPLEX64:     return _casinf(in1);
                default:            return _asin(in1);
            }
        case KP_ARCSINH:
            switch(etype) {
                case KP_COMPLEX128:    return _casinh(in1);
                case KP_COMPLEX64:     return _casinhf(in1);
                default:            return _asinh(in1);
            }
        case KP_ARCTAN:
            switch(etype) {
                case KP_COMPLEX128:    return _catan(in1);
                case KP_COMPLEX64:     return _catanf(in1);
                default:            return _atan(in1);
            }
        case KP_ARCTAN2:               return _atan2(in1, in2);
        case KP_ARCTANH:
            switch(etype) {
                case KP_COMPLEX128:    return _catanh(in1);
                case KP_COMPLEX64:     return _catanhf(in1);
                default:            return _atanh(in1);
            }
        case KP_BITWISE_AND:           return _bitw_and(in1, in2);
        case KP_BITWISE_OR:            return _bitw_or(in1, in2);
        case KP_BITWISE_XOR:           return _bitw_xor(in1, in2);
        case KP_CEIL:                  return _ceil(in1);
        case KP_COS:
            switch(etype) {
                case KP_COMPLEX128:    return _ccos(in1);
                case KP_COMPLEX64:     return _ccosf(in1);
                default:            return _cos(in1);
            }
        case KP_COSH:
            switch(etype) {
                case KP_COMPLEX128:    return _ccosh(in1);
                case KP_COMPLEX64:     return _ccoshf(in1);
                default:            return _cosh(in1);
            }
        case KP_DISCARD:               break;  // TODO: Raise exception
        case KP_DIVIDE:                return _div(in1, in2);
        case KP_EQUAL:                 return _eq(in1, in2);
        case KP_EXP:
            switch(etype) {
                case KP_COMPLEX128:    return _cexp(in1);
                case KP_COMPLEX64:     return _cexpf(in1);
                default:            return _exp(in1);
            }
        case KP_EXP2:
            switch(etype) {
                case KP_COMPLEX128:    return _cexp2(in1);
                case KP_COMPLEX64:     return _cexp2f(in1);
                default:            return _exp2(in1);
            }
        case KP_EXPM1:                 return _expm1(in1);
        case KP_EXTENSION_OPERATOR:    break;  // TODO: Raise exception
        case KP_FLOOD:                 break;  // TODO: Raise exception
        case KP_FLOOR:                 return _floor(in1);
        case KP_FREE:                  break;  // TODO: Raise exception
        case KP_GREATER:               return _gt(in1, in2);
        case KP_GREATER_EQUAL:         return _gteq(in1, in2);
        case KP_IDENTITY:              return in1;
        case KP_IMAG:
            switch(etype) {
                case KP_FLOAT32:       return _cimagf(in1);
                default:            return _cimag(in1);
            }
        case KP_INVERT:
            switch(etype) {
                case KP_BOOL:          return _invertb(in1);
                default:            return _invert(in1);
            }
        case KP_ISINF:                 return _isinf(in1);
        case KP_ISNAN:                 return _isnan(in1);
        case KP_LEFT_SHIFT:            return _bitw_leftshift(in1, in2);
        case KP_LESS:                  return _lt(in1, in2);
        case KP_LESS_EQUAL:            return _lteq(in1, in2);
        case KP_LOG:
            switch(etype) {
                case KP_COMPLEX128:    return _clog(in1);
                case KP_COMPLEX64:     return _clogf(in1);
                default:            return _log(in1);
            }
        case KP_LOG10:
            switch(etype) {
                case KP_COMPLEX128:    return _clog10(in1);
                case KP_COMPLEX64:     return _clog10f(in1);
                default:            return _log10(in1);
            }
        case KP_LOG1P:                 return _log1p(in1);
        case KP_LOG2:                  return _log2(in1);
        case KP_LOGICAL_AND:           return _logic_and(in1, in2);
        case KP_LOGICAL_NOT:           return _logic_not(in1);
        case KP_LOGICAL_OR:            return _logic_or(in1, in2);
        case KP_LOGICAL_XOR:           return _logic_xor(in1, in2);
        case KP_MAXIMUM:               return _max(in1, in2);
        case KP_MINIMUM:               return _min(in1, in2);
        case KP_MOD:                   return _mod(in1, in2);
        case KP_MULTIPLY:              return _mul(in1, in2);
        case KP_NONE:                  break;  // TODO: Raise exception
        case KP_NOT_EQUAL:             return _neq(in1, in2);
        case KP_POWER:
            switch(etype) {
                case KP_COMPLEX128:    return _cpow(in1, in2);
                case KP_COMPLEX64:     return _cpowf(in1, in2);
                default:            return _pow(in1, in2);
            }
        case KP_RANDOM:                return _random(in1, in2);
        case KP_RANGE:                 return _range();
        case KP_REAL:
            switch(etype) {
                case KP_FLOAT32:       return _crealf(in1);
                default:            return _creal(in1);
            }
        case KP_RIGHT_SHIFT:           return _bitw_rightshift(in1, in2);
        case KP_RINT:                  return _rint(in1);
        case KP_SIN:
            switch(etype) {
                case KP_COMPLEX128:    return _csin(in1);
                case KP_COMPLEX64:     return _csinf(in1);
                default:            return _sin(in1);
            }
        case KP_SIGN:
            switch(etype) {
                case KP_COMPLEX128:    return _div(in1, _parens(_add(_cabs(in1), _parens(_eq(in1, "0")))));
                case KP_COMPLEX64:     return _div(in1, _parens(_add(_cabsf(in1), _parens(_eq(in1, "0")))));
                default:            return _sub(
                                            _parens(_gt(in1, "0")),
                                            _parens(_lt(in1, "0"))
                                           );
            }

        case KP_SINH:
            switch(etype) {
                case KP_COMPLEX128:    return _csinh(in1);
                case KP_COMPLEX64:     return _csinhf(in1);
                default:            return _sinh(in1);
            }
        case KP_SQRT:
            switch(etype) {
                case KP_COMPLEX128:    return _csqrt(in1);
                case KP_COMPLEX64:     return _csqrtf(in1);
                default:            return _sqrt(in1);
            }
        case KP_SUBTRACT:              return _sub(in1, in2);
        case KP_SYNC:                  break;  // TODO: Raise exception
        case KP_TAN:
            switch(etype) {
                case KP_COMPLEX128:    return _ctan(in1);
                case KP_COMPLEX64:     return _ctanf(in1);
                default:            return _tan(in1);
            }
        case KP_TANH:
            switch(etype) {
                case KP_COMPLEX128:    return _ctanh(in1);
                case KP_COMPLEX64:     return _ctanhf(in1);
                default:            return _tanh(in1);
            }
        case KP_TRUNC:                 return _trunc(in1);
        default:                    return "NOT_IMPLEMENTED_YET";
    }
    return "NO NO< NO NO NO NO NONO NO NO NO NOTHERES NO LIMITS";
}

string Walker::synced_oper(KP_OPERATOR operation, KP_ETYPE etype, string out, string in1, string in2)
{
    stringstream ss;
    switch(operation) {
        case KP_MAXIMUM:
        case KP_MINIMUM:
        case KP_LOGICAL_AND:
        case KP_LOGICAL_OR:
        case KP_LOGICAL_XOR:
            ss << _omp_critical(_assign(out, oper(operation, etype, in1, in2)), "accusync");
            break;
        default:
            switch(etype) {
                case KP_COMPLEX64:
                case KP_COMPLEX128:
                    ss << _omp_critical(_assign(out, oper(operation, etype, in1, in2)), "accusync");
                    break;
                default:
                    ss << _omp_atomic(_assign(out, oper(operation, etype, in1, in2)));
                    break;
            }
            break;
    }
    return ss.str();
}

}}}
