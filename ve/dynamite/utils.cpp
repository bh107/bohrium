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
            return "unsigned char";
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
            return "complex float";
        case BH_COMPLEX128:
            return "complex double";
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
            return "z";
        case BH_INT8:
            return "b";
        case BH_INT16:
            return "s";
        case BH_INT32:
            return "i";
        case BH_INT64:
            return "l";
        case BH_UINT8:
            return "B";
        case BH_UINT16:
            return "S";
        case BH_UINT32:
            return "I";
        case BH_UINT64:
            return "L";
        case BH_FLOAT16:
            return "h";
        case BH_FLOAT32:
            return "f";
        case BH_FLOAT64:
            return "d";
        case BH_COMPLEX64:
            return "c";
        case BH_COMPLEX128:
            return "C";
        case BH_UNKNOWN:
            return "BH_UNKNOWN";
        default:
            return "Unknown type";
    }
}

const char* bhopcode_to_cexpr(bh_opcode opcode)
{
    switch(opcode) {
        case BH_ADD_REDUCE:
            //return "*a0_offset = *a0_offset + *a1_offset";
            return "*a0_offset += *tmp_offset";
        case BH_MULTIPLY_REDUCE:
            //return "*a0_offset = *a0_offset * *a1_offset";
            return "*a0_offset *= *a1_offset";
        case BH_MINIMUM_REDUCE:
            //return "*a0_offset = *a0_offset < *a1_offset ? *a0_offset : *a1_offset";
            return "*a0_offset = *a0_offset < *tmp_offset ? *a0_offset : *tmp_offset";
        case BH_MAXIMUM_REDUCE:
            //return "*a0_offset = *a0_offset < *a1_offset ? *a1_offset : *a0_offset";
            return "*a0_offset = *a0_offset < *tmp_offset ? *tmp_offset : *a0_offset";
        case BH_LOGICAL_AND_REDUCE:
            //return "*a0_offset = *a0_offset && *a1_offset";
            return "*a0_offset = *a0_offset && *tmp_offset";
        case BH_BITWISE_AND_REDUCE:
            return "*a0_offset &= *tmp_offset";
        case BH_LOGICAL_OR_REDUCE:
            return "*a0_offset = *a0_offset || *tmp_offset";
        case BH_BITWISE_OR_REDUCE:
            return "*a0_offset |= *tmp_offset";

        // Binary elementwise: ADD, MULTIPLY...
        case BH_ADD:
            return "*a0_offset = *a1_offset + *a2_offset";
        case BH_SUBTRACT:
            return "*a0_offset = *a1_offset - *a2_offset";
        case BH_MULTIPLY:
            return "*a0_offset = *a1_offset * *a2_offset";
        case BH_DIVIDE:
            return "*a0_offset = *a1_offset / *a2_offset";
        case BH_POWER:
            return "*a0_offset = pow( *a1_offset, *a2_offset )";
        case BH_GREATER:
            return "*a0_offset = *a1_offset > *a2_offset";
        case BH_GREATER_EQUAL:
            return "*a0_offset = *a1_offset >= *a2_offset";
        case BH_LESS:
            return "*a0_offset = *a1_offset < *a2_offset";
        case BH_LESS_EQUAL:
            return "*a0_offset = *a1_offset <= *a2_offset";
        case BH_EQUAL:
            return "*a0_offset = *a1_offset == *a2_offset";
        case BH_NOT_EQUAL:
            return "*a0_offset = *a1_offset != *a2_offset";
        case BH_LOGICAL_AND:
            return "*a0_offset = *a1_offset && *a2_offset";
        case BH_LOGICAL_OR:
            return "*a0_offset = *a1_offset || *a2_offset";
        case BH_LOGICAL_XOR:
            return "*a0_offset = (!*a1_offset != !*a2_offset)";
        case BH_MAXIMUM:
            return "*a0_offset = *a1_offset < *a2_offset ? *a2_offset : *a1_offset";
        case BH_MINIMUM:
            return "*a0_offset = *a1_offset < *a2_offset ? *a1_offset : *a2_offset";
        case BH_BITWISE_AND:
            return "*a0_offset = *a1_offset & *a2_offset";
        case BH_BITWISE_OR:
            return "*a0_offset = *a1_offset | *a2_offset";
        case BH_BITWISE_XOR:
            return "*a0_offset = *a1_offset ^ *a2_offset";
        case BH_LEFT_SHIFT:
            return "*a0_offset = (*a1_offset) << (*a2_offset)";
        case BH_RIGHT_SHIFT:
            return "*a0_offset = (*a1_offset) >> (*a2_offset)";
        case BH_ARCTAN2:
            return "*a0_offset = atan2( *a1_offset, *a2_offset )";
        case BH_MOD:
            return "*a0_offset = *a1_offset - floor(*a1_offset / *a2_offset) * *a2_offset";

        // Unary elementwise: SQRT, SIN...
        case BH_ABSOLUTE:
            return "*a0_offset = *a1_offset < 0.0 ? -*a1_offset: *a1_offset";
        case BH_LOGICAL_NOT:
            return "*a0_offset = !*a1_offset";
        case BH_INVERT:
            return "*a0_offset = ~*a1_offset";
        case BH_COS:
            return "*a0_offset = cos( *a1_offset )";
        case BH_SIN:
            return "*a0_offset = sin( *a1_offset )";
        case BH_TAN:
            return "*a0_offset = tan( *a1_offset )";
        case BH_COSH:
            return "*a0_offset = cosh( *a1_offset )";
        case BH_SINH:
            return "*a0_offset = sinh( *a1_offset )";
        case BH_TANH:
            return "*a0_offset = tanh( *a1_offset )";
        case BH_ARCSIN:
            return "*a0_offset = asin( *a1_offset )";
        case BH_ARCCOS:
            return "*a0_offset = acos( *a1_offset )";
        case BH_ARCTAN:
            return "*a0_offset = atan( *a1_offset )";
        case BH_ARCSINH:
            return "*a0_offset = asinh( *a1_offset )";
        case BH_ARCCOSH:
            return "*a0_offset = acosh( *a1_offset )";
        case BH_ARCTANH:
            return "*a0_offset = atanh( *a1_offset )";
        case BH_EXP:
            return "*a0_offset = exp( *a1_offset )";
        case BH_EXP2:
            return "*a0_offset = pow( 2, *a1_offset )";
        case BH_EXPM1:
            return "*a0_offset = expm1( *a1_offset )";
        case BH_LOG:
            return "*a0_offset = log( *a1_offset )";
        case BH_LOG2:
            return "*a0_offset = log2( *a1_offset )";
        case BH_LOG10:
            return "*a0_offset = log10( *a1_offset )";
        case BH_LOG1P:
            return "*a0_offset = log1p( *a1_offset )";
        case BH_SQRT:
            return "*a0_offset = sqrt( *a1_offset )";
        case BH_CEIL:
            return "*a0_offset = ceil( *a1_offset )";
        case BH_TRUNC:
            return "*a0_offset = trunc( *a1_offset )";
        case BH_FLOOR:
            return "*a0_offset = floor( *a1_offset )";
        case BH_RINT:
            return "*a0_offset = (*a1_offset > 0.0) ? floor(*a1_offset + 0.5) : ceil(*a1_offset - 0.5)";
        case BH_ISNAN:
            return "*a0_offset = isnan(*a1_offset)";
        case BH_ISINF:
            return "*a0_offset = isinf(*a1_offset)";
        case BH_IDENTITY:
            return "*a0_offset = *a1_offset";

        default:
            return "__UNKNOWN__";
    }
}


