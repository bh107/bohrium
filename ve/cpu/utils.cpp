#include "utils.hpp"
#include "thirdparty/MurmurHash3.h"

using namespace std;
namespace bohrium{
namespace utils{

const char TAG[] = "Utils";

/* Requires C++0x
std::string string_format(const std::string & fmt_str, ...)
{
    int final_n, n = fmt_str.size() * 2;    // reserve 2 times as much as the length of the fmt_str
    std::string str;
    std::unique_ptr<char[]> formatted;
    va_list ap;
    while(1) {
        formatted.reset(new char[n]);       // wrap the plain char array into the unique_ptr 
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
*/
std::string string_format(const std::string fmt_str, ...) {
    int size = 100;
    std::string str;
    va_list ap;
    while (1) {
        str.resize(size);
        va_start(ap, fmt_str);
        int n = vsnprintf((char *)str.c_str(), size, fmt_str.c_str(), ap);
        va_end(ap);
        if (n > -1 && n < size) {
            str.resize(n);
            return str;
        }
        if (n > -1) {
            size = n + 1;
        } else {
            size *= 2;
        }
    }
    return str;
}

/**
 *  Determines whether two operand have equivalent meta-data.
 *
 *  This function serves the same purpose as bh_view_identical, 
 *  but for tac-operands instead of bh_instruction.operand[...].
 *
 */
bool equivalent_operands(const operand_t& one, const operand_t& other)
{
    if (one.layout != other.layout) {
        return false;
    }
    if (one.data != other.data) {
        return false;
    }
    if (one.ndim != other.ndim) {
        return false;
    }
    if (one.start != other.start) {
        return false;
    }
    for(bh_intp j=0; j<one.ndim; ++j) {
        if (one.stride[j] != other.stride[j]) {
            return false;
        }
        if (one.shape[j] != other.shape[j]) {
            return false;
        }
    }
    return true;
}

/**
 *  Determines whether two operand have compatible meta-data.
 *
 *  This function serves the same purpose as bh_view_identical, 
 *  but for tac-operands instead of bh_instruction.operand[...].
 *
 */
bool compatible_operands(const operand_t& one, const operand_t& other)
{
    if ((one.layout == CONSTANT) || (other.layout == CONSTANT)) {
        return true;
    } 
    if (one.layout != other.layout) {
        return false;
    }
    if (one.ndim != other.ndim) {
        return false;
    }
    for(bh_intp j=0; j<one.ndim; ++j) {
        if (one.shape[j] != other.shape[j]) {
            return false;
        }
    }
    return true;
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
    ss << "{ op("<< operation_text(tac.op) << "(" << tac.op << ")),";
    ss << " oper(" << operator_text(tac.oper) << "(" << tac.oper << ")),";
    ss << " out("  << tac.out << "),";
    ss << " in1("  << tac.in1 << "),";
    ss << " in2("  << tac.in2 << ")";
    ss << " }";
    return ss.str();
}

uint32_t hash(std::string text)
{
    uint32_t seed = 4200;
    uint32_t hash[4];
    
    MurmurHash3_x86_128(text.c_str(), text.length(), seed, &hash);
    
    return hash[0];
}

string hash_text(std::string text)
{
    uint32_t hash[4];
    stringstream ss;

    uint32_t seed = 4200;

    MurmurHash3_x86_128(text.c_str(), text.length(), seed, &hash);
    ss << hash[0];
    ss << hash[1];
    ss << hash[2];
    ss << hash[3];
    
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

string tac_typesig_text(tac_t& tac, operand_t* scope)
{   
    stringstream ss;
    switch(tac_noperands(tac)) {
        case 3:
            ss << etype_text_shand(scope[tac.out].etype);
            ss << etype_text_shand(scope[tac.in1].etype);
            ss << etype_text_shand(scope[tac.in2].etype);
            break;
        case 2:
            ss << etype_text_shand(scope[tac.out].etype);
            ss << etype_text_shand(scope[tac.in1].etype);
            break;
        case 1:
            ss << etype_text_shand(scope[tac.out].etype);
            break;
        default:
            return string("");
    }
    return ss.str();
}

string tac_layout_text(tac_t& tac, operand_t* scope)
{
    stringstream ss;
    switch(tac_noperands(tac)) {
        case 3:
            ss << layout_text_shand(scope[tac.out].layout);
            ss << layout_text_shand(scope[tac.in1].layout);
            ss << layout_text_shand(scope[tac.in2].layout);
            break;
        case 2:
            ss << layout_text_shand(scope[tac.out].layout);
            ss << layout_text_shand(scope[tac.in1].layout);
            break;
        case 1:
            ss << layout_text_shand(scope[tac.out].layout);
            break;
        default:
            return string("");
    }
    return ss.str();
}

/**
 *  Write source-code to file.
 *  Filename will be along the lines of: kernel/<symbol>_<UID>.c
 *  NOTE: Does not overwrite existing files.
 */
bool write_file(string file_path, const char* sourcecode, size_t source_len)
{
    DEBUG(TAG, "write_file("<< file_path << ", ..., " << source_len << ");");

    int fd;              // Kernel file-descriptor
    FILE *fp = NULL;     // Handle for kernel-file
    const char *mode = "w";
    int err;

    fd = open(file_path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0644);
    if ((!fd) || (fd<1)) {
        err = errno;
        utils::error(err, "Engine::write_file [%s] in write_file(...).\n", file_path.c_str());
        return false;
    }
    fp = fdopen(fd, mode);
    if (!fp) {
        err = errno;
        utils::error(err, "fdopen(fildes= %d, flags= %s).", fd, mode);
        return false;
    }
    fwrite(sourcecode, 1, source_len, fp);
    fflush(fp);
    fclose(fp);
    close(fd);

    DEBUG(TAG, "write_file(...);");
    return true;
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
