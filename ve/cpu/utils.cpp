#include <iomanip>
#include "thirdparty/MurmurHash3.h"
#include "utils.hpp"

using namespace std;

namespace kp{
namespace core{

const char TAG[] = "Utils";

template <typename T>
string to_string(T val)
{
    stringstream stream;
    stream << val;
    return stream.str();
}

double get_const_value(const kp_operand & arg)
{
    switch(arg.etype) {
        case KP_BOOL:
            return (double)(*(unsigned char*)(arg.const_data));
        case KP_INT8:
            return (double)(*(int8_t*)(arg.const_data));
        case KP_INT16:
            return (double)(*(int16_t*)(arg.const_data));
        case KP_INT32:
            return (double)(*(int32_t*)(arg.const_data));
        case KP_INT64:
            return (double)(*(int64_t*)(arg.const_data));
        case KP_UINT8:
            return (double)(*(uint8_t*)(arg.const_data));
        case KP_UINT16:
            return (double)(*(uint16_t*)(arg.const_data));
        case KP_UINT32:
            return (double)(*(uint32_t*)(arg.const_data));
        case KP_UINT64:
            return (double)(*(uint64_t*)(arg.const_data));
        case KP_FLOAT32:
            return (double)(*(float*)(arg.const_data));
        case KP_FLOAT64:
            return (double)(*(double*)(arg.const_data));

        case KP_COMPLEX64:
        case KP_COMPLEX128:
        case KP_PAIRLL:
        default:
            throw invalid_argument(
                "Cannot get scalar-value of kp_operand with "
                "KP_ETYPE=[KP_COMPLEX64|KP_COMPLEX128|KP_PAIRLL]."
            );
    }
}

void set_const_value(const kp_operand & arg, double value)
{
    switch(arg.etype) {
        case KP_BOOL:
            (*(unsigned char*)(arg.const_data)) = (unsigned char)value;
            break;
        case KP_INT8:
            (*(int8_t*)(arg.const_data)) = (int8_t)value;
            break;
        case KP_INT16:
            (*(int16_t*)(arg.const_data)) = (int16_t)value;
            break;
        case KP_INT32:
            (*(int32_t*)(arg.const_data)) = (int32_t)value;
            break;
        case KP_INT64:
            (*(int64_t*)(arg.const_data)) = (int64_t)value;
            break;
        case KP_UINT8:
            (*(uint8_t*)(arg.const_data)) = (uint8_t)value;
            break;
        case KP_UINT16:
            (*(uint16_t*)(arg.const_data)) = (uint16_t)value;
            break;
        case KP_UINT32:
            (*(uint32_t*)(arg.const_data)) = (uint32_t)value;
            break;
        case KP_UINT64:
            (*(uint64_t*)(arg.const_data)) = (uint64_t)value;
            break;
        case KP_FLOAT32:
            (*(float*)(arg.const_data)) = (float)value;
            break;
        case KP_FLOAT64:
            (*(double*)(arg.const_data)) = (double)value;
            break;

        case KP_COMPLEX64:
        case KP_COMPLEX128:
        case KP_PAIRLL:
        default:
            throw invalid_argument(
                "Cannot set value of kp_operand with "
                "KP_ETYPE=[KP_COMPLEX64|KP_COMPLEX128|KP_PAIRLL]."
            );
            break;
    }
}

void tac_transform(kp_tac & tac, SymbolTable& symbol_table)
{
    switch(tac.op) {
        case KP_REDUCE_COMPLETE:
            if (symbol_table[tac.in1].layout == KP_SCALAR) {
                tac.op   = KP_MAP;
                tac.oper = KP_IDENTITY;
                tac.in2  = 0;
                goto transform_identity;
            }
            break;

        case KP_REDUCE_PARTIAL:
            if (symbol_table[tac.in1].layout == KP_SCALAR) {
                tac.op   = KP_MAP;
                tac.oper = KP_IDENTITY;
                tac.in2  = 0;
                goto transform_identity;
            } else if (symbol_table[tac.out].layout == KP_SCALAR) {
                tac.op = KP_REDUCE_COMPLETE;
            }
            break;

        case KP_SCAN:
            if (symbol_table[tac.in1].layout == KP_SCALAR) {
                tac.op   = KP_MAP;
                tac.oper = KP_IDENTITY;
                tac.in2  = 0;
                goto transform_identity;
            }
            break;

        case KP_ZIP:
            switch(tac.oper) {
                case KP_ADD:
                    if (((symbol_table[tac.in2].layout & (KP_SCALAR_CONST))>0) && \
                        (get_const_value(symbol_table[tac.in2]) == 0.0)) {
                        tac.op = KP_MAP;
                        tac.oper = KP_IDENTITY;
                        // tac.in1 = same as before
                        tac.in2 = 0;
                        goto transform_identity;
                    }
                    break;
                case KP_MULTIPLY:
                    if ((symbol_table[tac.in2].layout & (KP_SCALAR_CONST))>0) {
                        if (get_const_value(symbol_table[tac.in2]) == 0.0) {
                            tac.op = KP_MAP;
                            tac.oper = KP_IDENTITY;
                            tac.in1 = tac.in2;
                            set_const_value(symbol_table[tac.in1], 0);
                            tac.in2 = 0;
                        } else if (get_const_value(symbol_table[tac.in2]) == 1.0) {
                            tac.op = KP_MAP;
                            tac.oper = KP_IDENTITY;
                            // tac.in1 = same as before
                            tac.in2 = 0;
                            goto transform_identity;
                        }
                    }
                    break;
                case KP_DIVIDE:
                    if ((symbol_table[tac.in2].layout & (KP_SCALAR_CONST))>0) {
                        if (get_const_value(symbol_table[tac.in2]) == 1.0) {
                            tac.op = KP_MAP;
                            tac.oper = KP_IDENTITY;
                            // tac.in1 = same as before
                            tac.in2 = 0;
                            goto transform_identity;
                        }
                    }
                    break;
                default:
                    break;
            }
            break;

        case KP_MAP:
            switch(tac.oper) {
                case KP_IDENTITY:
                    transform_identity:
                    if (tac.out == tac.in1) {
                        tac.op = KP_NOOP;
                    }
                    break;
                default:
                    break;
            }
            break;

        default:
            break;
    }
}

bool equivalent(const kp_operand & one, const kp_operand & other)
{
    if (one.layout != other.layout) {
        return false;
    }
    if (one.layout == KP_SCALAR_CONST) {
        return false;
    }
    if (one.base != other.base) {
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



bool compatible(const kp_operand & one, const kp_operand & other)
{
    //
    // Scalar layouts are compatible with any other layout
    if (((one.layout & KP_SCALAR_LAYOUT)>0) || \
        ((other.layout & KP_SCALAR_LAYOUT)>0)) {
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

bool contiguous(const kp_operand & arg)
{
    int64_t weight = 1;
    for(int dim=arg.ndim-1; dim>=0; --dim) {
        if (arg.stride[dim] != weight) {
            return false;
        }
        weight *= arg.shape[dim];
    }
    return true;
}

string operand_access_text(const kp_operand & arg)
{
    /// Hmmm this is not entirely correct...
    // I forgot the simple thing:
    // A dim is strided when: stride[dim] != stride[dim+1]*shape[dim+1]
    bool is_strided[16] = {0};
    bool is_broadcast[16] = {0};
    int64_t stride_multipliers[16] = {0};

    int64_t weight = 1;
    int64_t n_strided_dims = 0;
    int64_t n_broadcast_dims = 0;
    for(int dim=arg.ndim-1; dim>=0; --dim) {
        if (arg.stride[dim] != weight) {
            is_strided[dim] = true;
            ++n_strided_dims;
            if(arg.stride[dim]) {
                int64_t stride_multiplier = max((int64_t)1, arg.stride[dim] / weight);
                stride_multipliers[dim] = stride_multiplier;
                weight *= stride_multiplier*arg.shape[dim];
            } else {
                stride_multipliers[dim] = 0;
                is_broadcast[dim] = true;
            }
        } else {
            stride_multipliers[dim] = 1;
            weight *= arg.shape[dim];
        }
    }

    stringstream ss, ss_strides, ss_broadcast, ss_multipliers;
    ss_strides << boolalpha;
    ss_broadcast << boolalpha;
    for(int dim=0; dim<arg.ndim; ++dim) {
        ss_strides << is_strided[dim];
        ss_broadcast << is_broadcast[dim];
        ss_multipliers << stride_multipliers[dim];
        if (dim!=arg.ndim-1) {
            ss_strides << ", ";
            ss_broadcast << ", ";
            ss_multipliers << ", ";
        }
    }
    ss << "Operand = "<< operand_text(arg) << endl;
    ss << "Multipliers        = " << ss_multipliers.str() << endl;
    ss << "Broadcoast dims(" << n_broadcast_dims << ") = " << ss_broadcast.str() << endl;
    ss << "Strided dims(" << n_strided_dims << ")    = " << ss_strides.str() << endl;
    return ss.str();
}

KP_LAYOUT determine_layout(const kp_operand & arg)
{
    const int64_t inner_dim = arg.ndim-1;
    
    // KP_CONSECUTIVE: stride[dim] == stride[dim+1]*shape[dim+1]
    // KP_CONTIGUOUS:  stride[dim] == stride[dim+1]*shape[dim+1] and stride[inner] == 1
    bool consecutive = true;    
    int64_t weight = arg.stride[inner_dim];
    int64_t nelements = 1;
    for(int dim=inner_dim; dim>=0; --dim) {
        if (arg.stride[dim] != weight) {
            consecutive = false;
        }
        nelements *= arg.shape[dim];
        weight = arg.shape[dim]*arg.stride[dim];
    }

    if (nelements == 1) {
        return KP_SCALAR;
    } else if (consecutive and arg.stride[inner_dim] == 1) {
        return KP_CONTIGUOUS;
    } else if (consecutive and arg.stride[inner_dim] > 1) {
        return KP_CONSECUTIVE;
    } else {
        return KP_STRIDED;
    }
}

std::string iterspace_text(const kp_iterspace & iterspace)
{
    stringstream ss_layout_ndim, ss_shape, ss_nelem, ss;

    ss_layout_ndim << setfill('-');
    ss_layout_ndim << left;
    ss_layout_ndim << setw(14);
    ss_layout_ndim << core::layout_text(iterspace.layout);
    ss_layout_ndim << setw(4);
    ss_layout_ndim << " "+ to_string(iterspace.ndim) + "D ";

    for(int64_t dim=0; dim <iterspace.ndim; ++dim) {
        //ss_shape << iterspace.shape[dim];
        ss_shape << iterspace.shape[dim];
        if (dim!=iterspace.ndim-1) {
            ss_shape << "x";
        }
    }

    ss_nelem << " #";
    ss_nelem << setw(16);
    ss_nelem << setfill('-');
    ss_nelem << right;
    ss_nelem << to_string(iterspace.nelem);
    
    ss << ss_layout_ndim.str();
    ss << setfill('-');
    ss << right;
    ss << setw(21); 
    ss << ss_shape.str();
    ss << ss_nelem.str();
 
    return ss.str();
}

std::string operand_text(const kp_operand & operand)
{
    stringstream ss;
    ss << "{";
    ss << " layout("    << core::layout_text(operand.layout) << "),";
    ss << " nelem("     << operand.nelem << "),";
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
    ss << "}";

    return ss.str();
}

std::string omask_aop_text(uint32_t omask)
{
    stringstream ss;
    std::vector<std::string> entries;
    for(uint32_t op= KP_MAP; op<= KP_NOOP; op=op<<1) {
        if ((((omask&op)>0) and ((op& KP_ARRAY_OPS)>0))) {
            entries.push_back(operation_text((KP_OPERATION)op));
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
    for(uint32_t op= KP_MAP; op<= KP_NOOP; op=op<<1) {
        if((omask&op)>0) {
            entries.push_back(operation_text((KP_OPERATION)op));
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

std::string tac_text(const kp_tac & tac)
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

string tac_text(const kp_tac & tac, SymbolTable& symbol_table)
{
    std::stringstream ss;
    ss << "{ op("<< operation_text(tac.op) << "(" << tac.op << ")),";
    ss << " oper(" << operator_text(tac.oper) << "(" << tac.oper << ")),";
    switch(tac_noperands(tac)) {
        case 3:
            ss << endl;
            ss << " in2("  << tac.in2 << ") = ";
            ss << operand_text(symbol_table[tac.in2]);
            ss << ")";
        case 2:
            ss << endl;
            ss << " in1("  << tac.in1 << ") = ";
            ss << operand_text(symbol_table[tac.in1]);
            ss << ")";
        case 1:
            ss << endl;
            ss << " out("  << tac.out << ") = ";
            ss << operand_text(symbol_table[tac.out]);
            ss << ")";
            break;
    }
    ss << endl;
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

size_t tac_noperands(const kp_tac & tac)
{
	return kp_tac_noperands(&tac);
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
