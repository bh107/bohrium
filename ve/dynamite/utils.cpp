#include <fstream>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include "bh.h"

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

const char* bhtype_to_ctype(bh_type type)
{
    switch(type) {
        case BH_BOOL:
            return "bool";
        case BH_INT8:
            return "int8_t";
        case BH_INT16:
            return "int16_t";
        case BH_INT32:
            return "int32_t";
        case BH_INT64:
            return "int64_t";
        case BH_UINT8:
            return "uint8_t";
        case BH_UINT16:
            return "uint16_t";
        case BH_UINT32:
            return "uint32_t";
        case BH_UINT64:
            return "uint64_t";
        case BH_FLOAT16:
            return "uint16_t";
        case BH_FLOAT32:
            return "float";
        case BH_FLOAT64:
            return "double";
        case BH_COMPLEX64:
            return "struct {float real, imag; }";
        case BH_COMPLEX128:
            return "struct {double real, imag; }";
        case BH_UNKNOWN:
            return "BH_UNKNOWN";
        default:
            return "Unknown type";
    }
}

const char* bhtype_to_shorthand(bh_type type)
{
    switch(type) {
        case BH_BOOL:
            return "B";
        case BH_INT8:
            return "I8";
        case BH_INT16:
            return "I16";
        case BH_INT32:
            return "I32";
        case BH_INT64:
            return "I64";
        case BH_UINT8:
            return "U8";
        case BH_UINT16:
            return "U16";
        case BH_UINT32:
            return "U32";
        case BH_UINT64:
            return "U64";
        case BH_FLOAT16:
            return "U16";
        case BH_FLOAT32:
            return "F";
        case BH_FLOAT64:
            return "D";
        case BH_COMPLEX64:
            return "CF";
        case BH_COMPLEX128:
            return "CD";
        case BH_UNKNOWN:
            return "UK";
        default:
            return "UK";
    }
}

const char* bhopcode_to_csrc(bh_opcode opc) {
    switch(opc) {
        case BH_ADD:
            return "+";
        case BH_SUBTRACT:
            return "-";
        case BH_MULTIPLY:
            return "*";
        case BH_DIVIDE:
            return "/";
        case BH_MOD:
            return "%";
        case BH_BITWISE_AND:
            return "&";
        case BH_BITWISE_OR:
            return "|";
        case BH_BITWISE_XOR:
            return "^";
        case BH_LEFT_SHIFT:
            return "<<";
        case BH_RIGHT_SHIFT:
            return ">>";
        case BH_EQUAL:
            return "==";
        case BH_NOT_EQUAL:
            return "!=";
        case BH_GREATER:
            return ">";
        case BH_GREATER_EQUAL:
            return ">=";
        case BH_LESS:
            return "<";
        case BH_LESS_EQUAL:
            return "<=";
        case BH_LOGICAL_AND:
            return "&&";
        case BH_LOGICAL_OR:
            return "||";
        default:
            return "{{UNKNOWN_OPCODE}}";
    }
}

