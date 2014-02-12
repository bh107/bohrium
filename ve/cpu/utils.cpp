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
bool is_contiguous(bh_kernel_arg_t* arg)
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

int noperands(OPERATION op, OPERATOR oper)
{
    switch(op) {
        case EWISE_U:
            return 2;
        case EWISE_B:
            return 3;
        case SCAN:
            return 3;
        case REDUCE:
            return 3;
        case GENERATOR:
            switch(oper) {
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
            switch(oper) {
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
}

int layoutmask(bytecode_t* bytecode, bh_args_t* args)
{
    switch(noperands(bytecode)) {
        case 3:
            return args[bytecode->out].layoutmask | args[bytecode->in1].layoutmask | args[bytecode->in2].layoutmask;
        case 2:
            return args[bytecode->out].layoutmask | args[bytecode->in1].layoutmask;
        case 1:
            return args[bytecode->out].layoutmask;
    }
    return mask;
}
