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

const char* type_text(bh_type type)
{
    switch(type)
    {
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

