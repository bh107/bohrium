#include <iomanip>
#include "utils.hpp"
#include "thirdparty/MurmurHash3.h"

using namespace std;
namespace bohrium{
namespace core{

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
    if ((one.layout == SCALAR_CONST)    && \
        (one.etype == other.etype)) {
        switch(one.etype) {
            case BOOL:
                return (*(unsigned char*)(*(one.data))  == \
                        *(unsigned char*)(*(other.data)));
            case INT8:
                return (*(int8_t*)(*(one.data)) == \
                        *(int8_t*)(*(other.data)));
            case INT16:
                return (*(int16_t*)(*(one.data)) == \
                        *(int16_t*)(*(other.data)));
            case INT32:
                return (*(int32_t*)(*(one.data)) == \
                        *(int32_t*)(*(other.data)));
            case INT64:
                return (*(int64_t*)(*(one.data)) == \
                        *(int64_t*)(*(other.data)));
            case UINT8:
                return (*(uint8_t*)(*(one.data)) == \
                        *(uint8_t*)(*(other.data)));
            case UINT16:
                return (*(uint16_t*)(*(one.data)) == \
                        *(uint16_t*)(*(other.data)));
            case UINT32:
                return (*(uint32_t*)(*(one.data)) == \
                        *(uint32_t*)(*(other.data)));
            case UINT64:
                return (*(uint64_t*)(*(one.data)) == \
                        *(uint64_t*)(*(other.data)));
            case FLOAT32:
                return (*(float*)(*(one.data)) == \
                        *(float*)(*(other.data)));
            case FLOAT64:
                return (*(double*)(*(one.data)) == \
                        *(double*)(*(other.data)));
            case COMPLEX64:
            case COMPLEX128:
            case PAIRLL:
                return false;
        }
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
    if (one.base != other.base) {
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
    if (one.start != other.start) {
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

bool contiguous(const operand_t& arg)
{
    int64_t shape = 1;
    for(int ldim=arg.ndim-1; ldim>=0; --ldim) {
        if (arg.stride[ldim] != shape) {
            return false;
        }
        shape *= arg.shape[ldim];
    }
    return true;
}

std::string iterspace_text(const iterspace_t& iterspace)
{
    stringstream ss;
    ss << setw(12);
    ss << setfill('-');
    ss << core::layout_text(iterspace.layout) << "_";
    ss << iterspace.ndim << "D_";

    stringstream ss_shape;
    for(int64_t dim=0; dim <iterspace.ndim; ++dim) {
        ss_shape << iterspace.shape[dim];
        if (dim!=iterspace.ndim-1) {
            ss_shape << "x";
        }
    }
    ss << left;
    ss << setw(20);
    ss << ss_shape.str();
    ss << "_";
    ss << iterspace.nelem;
    
    return ss.str();
}

std::string operand_text(const operand_t& operand)
{
    stringstream ss;
    ss << "{";
    ss << " layout("    << core::layout_text(operand.layout) << "),";
    ss << " nelem("     << operand.nelem << "),";
    ss << " data("      << *(operand.data) << "),";
    ss << " const_data("<< operand.const_data << "),";
    ss << " etype("     << core::etype_text(operand.etype) << "),";
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

std::string omask_aop_text(uint32_t omask)
{
    stringstream ss;
    std::vector<std::string> entries;
    for(uint32_t op=MAP; op<=NOOP; op=op<<1) {
        if ((((omask&op)>0) and ((op&ARRAY_OPS)>0))) {
            entries.push_back(operation_text((OPERATION)op));
        }
    }
    for(std::vector<std::string>::iterator eit=entries.begin();
        eit!=entries.end();
        ++eit) {
        ss << *eit;
        eit++;
        if (eit!=entries.end()) {
            ss << "|";
        }
        eit--;
    }
    return ss.str();
}

std::string omask_text(uint32_t omask)
{
    stringstream ss;
    std::vector<std::string> entries;
    for(uint32_t op=MAP; op<=NOOP; op=op<<1) {
        if((omask&op)>0) {
            entries.push_back(operation_text((OPERATION)op));
        }
    }
    for(std::vector<std::string>::iterator eit=entries.begin();
        eit!=entries.end();
        ++eit) {
        ss << *eit;
        eit++;
        if (eit!=entries.end()) {
            ss << " | ";
        }
        eit--;
    }
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
    ss << std::hex;
    ss << std::setw(8);
    ss << std::setfill('0');
    ss << hash[0];
    ss << "_";
    ss << std::setw(8);
    ss << std::setfill('0');
    ss << hash[1];
    ss << "_";
    ss << std::setw(8);
    ss << std::setfill('0');
    ss << hash[2];
    ss << "_";
    ss << std::setw(8);
    ss << std::setfill('0');
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

bool write_file(string file_path, const char* sourcecode, size_t source_len)
{
    int fd;              // Kernel file-descriptor
    FILE *fp = NULL;     // Handle for kernel-file
    const char *mode = "w";
    int err;

    fd = open(file_path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0644);
    if ((!fd) || (fd<1)) {
        err = errno;
        core::error(err, "Engine::write_file [%s] in write_file(...).\n", file_path.c_str());
        return false;
    }
    fp = fdopen(fd, mode);
    if (!fp) {
        err = errno;
        core::error(err, "fdopen(fildes= %d, flags= %s).", fd, mode);
        return false;
    }
    fwrite(sourcecode, 1, source_len, fp);
    fflush(fp);
    fclose(fp);
    close(fd);

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
