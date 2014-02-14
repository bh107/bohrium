#include <fstream>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include "bh.h"
#include "bh_ve_cpu.h"

#include "utils.auto.cpp"

void bh_string_option(char *&option, const char *env_name, const char *conf_name)
{
    option = getenv(env_name);           // For the compiler
    if (NULL==option) {
        option = bh_component_config_lookup(&myself, conf_name);
    }
    char err_msg[100];

    if (!option) {
        sprintf(err_msg, "cpu-ve: String is not set; option (%s).\n", conf_name);
        throw runtime_error(err_msg);
    }
}

void bh_path_option(char *&option, const char *env_name, const char *conf_name)
{
    option = getenv(env_name);           // For the compiler
    if (NULL==option) {
        option = bh_component_config_lookup(&myself, conf_name);
    }
    char err_msg[100];

    if (!option) {
        sprintf(err_msg, "cpu-ve: Path is not set; option (%s).\n", conf_name);
        throw runtime_error(err_msg);
    }
    if (0 != access(option, F_OK)) {
        if (ENOENT == errno) {
            sprintf(err_msg, "cpu-ve: Path does not exist; path (%s).\n", option);
        } else if (ENOTDIR == errno) {
            sprintf(err_msg, "cpu-ve: Path is not a directory; path (%s).\n", option);
        } else {
            sprintf(err_msg, "cpu-ve: Path is broken somehow; path (%s).\n", option);
        }
        throw runtime_error(err_msg);
    }
}

/**
 * Read the entire file provided via filename into memory.
 *
 * It is the resposibility of the caller to de-allocate the buffer.
 *
 * @return size_t bytes read.
 */
size_t read_file(const char* filename, char** contents)
{
    int size = 0;

    std::ifstream file(filename, std::ios::in|std::ios::binary|std::ios::ate);

    if (file.is_open()) {
        size = file.tellg();
        *contents = (char*)malloc(size);
        file.seekg(0, std::ios::beg);
        file.read(*contents, size);
        file.close();
    }

    return size;
}

void assign_string(char*& output, const char* input)
{
    size_t length = strlen(input);

    output = (char*)malloc(sizeof(char) * length+1);
    if (!output) {
        std::cout << "Something went terribly wrong!" << std::endl;
    }
    strncpy(output, input, length);
    output[length] = '\0';
}

/**
 *  Determine whether or not a kernel argument i contiguous or strided.
 */
bool is_contiguous(block_arg_t* arg)
{
    if ((arg->ndim == 3) && \
        (arg->stride[2] == 1) && \
        (arg->stride[1] == arg->shape[2]) && \
        (arg->stride[0] == arg->shape[2]*arg->shape[1])
    ) {
        return true;
    } else if ((arg->ndim == 2) && \
               (arg->stride[1] == 1) && \
               (arg->stride[0] == arg->shape[1])) {
        return true;
    } else if ((arg->ndim == 1) && (arg->stride[0] == 1)) {
        return true;
    }

    return false;
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

std::string tac_text(tac_t* tac)
{
    std::stringstream ss;
    ss << "[op="<< operation_text(tac->op) << "(" << tac->op << ")";
    ss << ", oper=" << tac->oper;
    ss << ", out=" << tac->out;
    ss << ", in1=" << tac->in1;
    ss << ", in2=" << tac->in2;
    ss << "]" << endl;
    return ss.str();
}

std::string block_text(block_t* block)
{
    std::stringstream ss;
    ss << "block(";
    ss << "length=" << std::to_string(block->length);
    ss << ", nargs=" << block->nargs;
    ss << ", omask=" << block->omask;
    ss << ") {" << endl;
    for(int i=0; i<block->length; ++i) {
        ss << "  " << tac_text(&block->program[i]);
    }
    ss << "  (" << block->symbol << ")" << endl;
    ss << "}";
    
    return ss.str();
}

std::string layout_text_short(LAYOUT layout)
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

const int noperands(tac_t* tac)
{
    switch(tac->op) {
        case MAP:
            return 2;
        case ZIP:
            return 3;
        case SCAN:
            return 3;
        case REDUCE:
            return 3;
        case GENERATE:
            switch(tac->oper) {
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
            switch(tac->oper) {
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

std::string bh_type_text_short(bh_type type)
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

std::string tac_typesig_text(tac_t* tac, block_arg_t* scope)
{
    switch(noperands(tac)) {
        case 3:
            return  bh_type_text_short(scope[tac->out].type)+\
                    bh_type_text_short(scope[tac->in1].type)+\
                    bh_type_text_short(scope[tac->in2].type);
        case 2:
            return  bh_type_text_short(scope[tac->out].type)+\
                    bh_type_text_short(scope[tac->in1].type);
        case 1:
            return  bh_type_text_short(scope[tac->out].type);
        default:
            return "";
    }
}

std::string tac_layout_text(tac_t* tac, block_arg_t* scope)
{
    switch(noperands(tac)) {
        case 3:
            return  layout_text_short(scope[tac->out].layout)+\
                    layout_text_short(scope[tac->in1].layout)+\
                    layout_text_short(scope[tac->in2].layout);
        case 2:
            return  layout_text_short(scope[tac->out].layout)+\
                    layout_text_short(scope[tac->in1].layout);
        case 1:
            return  layout_text_short(scope[tac->out].layout);
        default:
            return "";
    }
}
