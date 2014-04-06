#include "utils.hpp"
#include "thirdparty/MurmurHash3.h"

using namespace std;
namespace bohrium{
namespace utils{

const char TAG[] = "Utils";

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

bool equivalent(const operand_t& one, const operand_t& other)
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

bool compatible(const operand_t& one, const operand_t& other)
{
    //
    // Scalar layouts are compatible with any other layout
    if (((one.layout & SCALAR_LAYOUT)>0) || \
        ((other.layout & SCALAR_LAYOUT)>0)) {
        return true;
    }

    /*
    //
    // Array layouts of different types are not compatible 
    if (one.layout != other.layout) {
        return false;
    }*/
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

bool contiguous(const operand_t& arg)
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

std::string operand_text(const operand_t& operand)
{
    stringstream ss;
    ss << "{";
    ss << " layout("    << utils::layout_text(operand.layout) << "),";
    ss << " nelem("     << operand.nelem << "),";
    ss << " data("      << *(operand.data) << "),";
    ss << " const_data("<< operand.const_data << "),";
    ss << " etype("     << utils::etype_text(operand.etype) << "),";
    ss << " ndim("      << operand.ndim << "),";
    ss << " start("     << operand.start << "),";        
    ss << " shape(";
    for(int64_t dim=0; dim < operand.ndim; ++dim) {
        ss << operand.shape[dim];
        if (dim != (operand.ndim-1)) {
            ss << ", ";
        }
    }
    ss << "),";
    ss << " stride(";
    for(int64_t dim=0; dim < operand.ndim; ++dim) {
        ss << operand.stride[dim];
        if (dim != (operand.ndim-1)) {
            ss << ", ";
        }
    }
    ss << ") ";
    ss << "}" << endl;

    return ss.str();
}

std::string tac_text(const tac_t& tac)
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

int tac_noperands(const tac_t& tac)
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
                    return 2;
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
