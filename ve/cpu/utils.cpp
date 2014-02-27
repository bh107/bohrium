#include "utils.hpp"

using namespace std;
namespace bohrium{
namespace utils{

std::string string_format(const std::string & fmt_str, ...)
{
    int final_n, n = fmt_str.size() * 2; /* reserve 2 times as much as the length of the fmt_str */
    std::string str;
    std::unique_ptr<char[]> formatted;
    va_list ap;
    while(1) {
        formatted.reset(new char[n]); /* wrap the plain char array into the unique_ptr */
        strcpy(&formatted[0], fmt_str.c_str());
        va_start(ap, fmt_str);
        final_n = vsnprintf(&formatted[0], n, fmt_str.c_str(), ap);
        va_end(ap);
        if (final_n < 0 || final_n >= n)
            n += abs(final_n - n + 1);
        else
            break;
    }
    return std::string(formatted.get());
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
    ss << ", oper=" << operator_text(tac.oper) << "(" << tac.oper << ")";
    ss << ", out="  << tac.out;
    ss << ", in1="  << tac.in1;
    ss << ", in2="  << tac.in2;
    ss << "]" << endl;
    return ss.str();
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
        case NOOP:
            return 0;
    }
    return 0;
}

std::string tac_typesig_text(tac_t& tac, operand_t* scope)
{
    switch(tac_noperands(tac)) {
        case 3:
            return  etype_text_shand(scope[tac.out].type)+\
                    etype_text_shand(scope[tac.in1].type)+\
                    etype_text_shand(scope[tac.in2].type);
        case 2:
            return  etype_text_shand(scope[tac.out].type)+\
                    etype_text_shand(scope[tac.in1].type);
        case 1:
            return  etype_text_shand(scope[tac.out].type);
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
