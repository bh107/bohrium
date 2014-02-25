#include "utils.hpp"

using namespace std;
namespace bohrium{
namespace utils{

std::string bh_type_text_shand(bh_type type)
{
    switch(type) {
        case BH_BOOL: return "z";
        case BH_INT8: return "b";
        case BH_INT16: return "s";
        case BH_INT32: return "i";
        case BH_INT64: return "l";
        case BH_UINT8: return "B";
        case BH_UINT16: return "S";
        case BH_UINT32: return "I";
        case BH_UINT64: return "L";
        case BH_FLOAT32: return "f";
        case BH_FLOAT64: return "d";
        case BH_COMPLEX64: return "c";
        case BH_COMPLEX128: return "C";
        case BH_R123: return "R";
        case BH_UNKNOWN: return "U";
        default:
            return "{{UNKNOWN}}";
    }
}

std::string operation_text(OPERATION op)
{
    switch(op) {
        case MAP:
            return "MAP";
        case ZIP:
            return "ZIP";
        case SCAN:
            return "SCAN";
        case REDUCE:
            return "REDUCE";
        case GENERATE:
            return "GENERATE";
        case SYSTEM:
            return "SYSTEM";
        case EXTENSION:
            return "EXTENSION";
        default:
            return "_ERR_";
    }
}

// TODO: implement
std::string operator_text(OPERATOR op)
{
    return "IMPLEMENT_ME";
}

std::string etype_text(ETYPE type)
{
    return "IMPLEMENT_ME";
}

std::string etype_text_shand(ETYPE type)
{
    return "IMPLEMENT_ME";
}

ETYPE bhtype_to_etype(bh_type type)
{
    return FLOAT32;
}

/**
 *  Determine whether an operand has a contiguous layout.
 */
bool is_contiguous(operand_t& arg)
{
    if ((arg.ndim == 3) && \
        (arg.stride[2] == 1) && \
        (arg.stride[1] == arg.shape[2]) && \
        (arg.stride[0] == arg.shape[2]*arg.shape[1])
    ) {
        return true;
    } else if ((arg.ndim == 2) && \
               (arg.stride[1] == 1) && \
               (arg.stride[0] == arg.shape[1])) {
        return true;
    } else if ((arg.ndim == 1) && (arg.stride[0] == 1)) {
        return true;
    }

    return false;
}

std::string tac_text(tac_t& tac)
{
    std::stringstream ss;
    ss << "[op="<< operation_text(tac.op) << "(" << tac.op << ")";
    ss << ", oper=" << tac.oper;
    ss << ", out="  << tac.out;
    ss << ", in1="  << tac.in1;
    ss << ", in2="  << tac.in2;
    ss << "]" << endl;
    return ss.str();
}

std::string layout_text_shand(LAYOUT layout)
{
    switch(layout) {
        case CONSTANT:
            return "K";
        case CONTIGUOUS:
            return "C";
        case STRIDED:
            return "S";
        case SPARSE:
            return "P";
    }
    return "ERR";
}

int tac_noperands(tac_t& tac)
{
    switch(tac.op) {
        case MAP:
            return 2;
        case ZIP:
            return 3;
        case SCAN:
            return 3;
        case REDUCE:
            return 3;
        case GENERATE:
            switch(tac.oper) {
                case FLOOD:
                    return 2;
                case RANDOM:
                    return 3;
                case RANGE:
                    return 1;
                default:
                    throw runtime_error("noperands does not know how many operands are used.");
                    return 0;
            }
        case SYSTEM:
            switch(tac.oper) {
                case DISCARD:
                case FREE:
                case SYNC:
                    return 1;
                case NONE:
                    return 0;
                default:
                    throw runtime_error("noperands does not know how many operands are used.");
                    return 0;
            }
            break;
        case EXTENSION:
            return 3;
    }
    return 0;
}

std::string tac_typesig_text(tac_t& tac, operand_t* scope)
{
    switch(tac_noperands(tac)) {
        case 3:
            return  bh_type_text_shand(scope[tac.out].type)+\
                    bh_type_text_shand(scope[tac.in1].type)+\
                    bh_type_text_shand(scope[tac.in2].type);
        case 2:
            return  bh_type_text_shand(scope[tac.out].type)+\
                    bh_type_text_shand(scope[tac.in1].type);
        case 1:
            return  bh_type_text_shand(scope[tac.out].type);
        default:
            return "";
    }
}

std::string tac_layout_text(tac_t& tac, operand_t* scope)
{
    switch(tac_noperands(tac)) {
        case 3:
            return  layout_text_shand(scope[tac.out].layout)+\
                    layout_text_shand(scope[tac.in1].layout)+\
                    layout_text_shand(scope[tac.in2].layout);
        case 2:
            return  layout_text_shand(scope[tac.out].layout)+\
                    layout_text_shand(scope[tac.in1].layout);
        case 1:
            return  layout_text_shand(scope[tac.out].layout);
        default:
            return "";
    }
}

//
//  MOVE THESE TO CORE
//

// Create nice error-messages...
int error(int errnum, const char *fmt, ...)
{
    va_list va;
    int ret;

    char err_msg[500];
    sprintf(err_msg, "Error[%d, %s] from: %s", errnum, strerror(errnum), fmt);
    va_start(va, fmt);
    ret = vfprintf(stderr, err_msg, va);
    va_end(va);
    return ret;
}

int error(const char *err_msg, const char *fmt, ...)
{
    va_list va;
    int ret;

    char err_txt[500];
    sprintf(err_txt, "Error[%s] from: %s", err_msg, fmt);
    va_start(va, fmt);
    ret = vfprintf(stderr, err_txt, va);
    va_end(va);
    return ret;
}

}}
