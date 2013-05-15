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
        /* For now only element-wise wise

        // System (memory and stuff)
        case BH_DISCARD:
            return "forget(*off0)";
        case BH_FREE:
            return "free(*off0)";
        case BH_SYNC:
            return "SYNC";
        case BH_NONE:
            return "No *offeration.";

        // Extensions (ufuncs)
        case BH_USERFUNC:
            return "USER DEFINED BEHAVIOR";

        // Partial Reductions
        case BH_ADD_REDUCE:
            return "sum(a, axis)";
        case BH_MULTIPLY_REDUCE:
            return "product(a, axis)";
        case BH_MINIMUM_REDUCE:
            return "min(a, axis)";
        case BH_MAXIMUM_REDUCE:
            return "max(a, axis)";
        case BH_LOGICAL_AND_REDUCE:
            return "all(a, axis)";
        case BH_BITWISE_AND_REDUCE:
            return "all(a, axis)";
        case BH_LOGICAL_OR_REDUCE:
            return "any(a, axis)";
        case BH_BITWISE_OR_REDUCE:
            return "any(a, axis)";
        */

        // Binary elementwise: ADD, MULTIPLY...
        case BH_ADD:
            return "*off0 = *off1 + *off2";
        case BH_SUBTRACT:
            return "*off0 = *off1 - *off2";
        case BH_MULTIPLY:
            return "*off0 = *off1 * *off2";
        case BH_DIVIDE:
            return "*off0 = *off1 / *off2";
        case BH_POWER:
            return "*off0 = pow( *off1, *off2 )";
        case BH_GREATER:
            return "*off0 = *off1 > *off2";
        case BH_GREATER_EQUAL:
            return "*off0 = *off1 >= *off2";
        case BH_LESS:
            return "*off0 = *off1 < *off2";
        case BH_LESS_EQUAL:
            return "*off0 = *off1 <= *off2";
        case BH_EQUAL:
            return "*off0 = *off1 == *off2";
        case BH_NOT_EQUAL:
            return "*off0 = *off1 != *off2";
        case BH_LOGICAL_AND:
            return "*off0 = *off1 && *off2";
        case BH_LOGICAL_OR:
            return "*off0 = *off1 || *off2";
        case BH_LOGICAL_XOR:
            return "*off0 = (!*off1 != !*off2)";
        case BH_MAXIMUM:
            return "*off0 = *off1 < *off2 ? *off2 : *off1";
        case BH_MINIMUM:
            return "*off0 = *off1 < *off2 ? *off1 : *off2";
        case BH_BITWISE_AND:
            return "*off0 = *off1 & *off2";
        case BH_BITWISE_OR:
            return "*off0 = *off1 | *off2";
        case BH_BITWISE_XOR:
            return "*off0 = *off1 ^ *off2";
        case BH_LEFT_SHIFT:
            return "*off0 = (*off1) << (*off2)";
        case BH_RIGHT_SHIFT:
            return "*off0 = (*off1) >> (*off2)";
        case BH_ARCTAN2:
            return "*off0 = atan2( *off1, *off2 )";
        case BH_MOD:
            return "*off0 = *off1 - floor(*off1 / *off2) * *off2";

        // Unary elementwise: SQRT, SIN...
        case BH_ABSOLUTE:
            return "*off0 = *off1 < 0.0 ? -*off1: *off1";
        case BH_LOGICAL_NOT:
            return "*off0 = !*off1";
        case BH_INVERT:
            return "*off0 = ~*off1";
        case BH_COS:
            return "*off0 = cos( *off1 )";
        case BH_SIN:
            return "*off0 = sin( *off1 )";
        case BH_TAN:
            return "*off0 = tan( *off1 )";
        case BH_COSH:
            return "*off0 = cosh( *off1 )";
        case BH_SINH:
            return "*off0 = sinh( *off1 )";
        case BH_TANH:
            return "*off0 = tanh( *off1 )";
        case BH_ARCSIN:
            return "*off0 = asin( *off1 )";
        case BH_ARCCOS:
            return "*off0 = acos( *off1 )";
        case BH_ARCTAN:
            return "*off0 = atan( *off1 )";
        case BH_ARCSINH:
            return "*off0 = asinh( *off1 )";
        case BH_ARCCOSH:
            return "*off0 = acosh( *off1 )";
        case BH_ARCTANH:
            return "*off0 = atanh( *off1 )";
        case BH_EXP:
            return "*off0 = exp( *off1 )";
        case BH_EXP2:
            return "*off0 = pow( 2, *off1 )";
        case BH_EXPM1:
            return "*off0 = expm1( *off1 )";
        case BH_LOG:
            return "*off0 = log( *off1 )";
        case BH_LOG2:
            return "*off0 = log2( *off1 )";
        case BH_LOG10:
            return "*off0 = log10( *off1 )";
        case BH_LOG1P:
            return "*off0 = log1p( *off1 )";
        case BH_SQRT:
            return "*off0 = sqrt( *off1 )";
        case BH_CEIL:
            return "*off0 = ceil( *off1 )";
        case BH_TRUNC:
            return "*off0 = trunc( *off1 )";
        case BH_FLOOR:
            return "*off0 = floor( *off1 )";
        case BH_RINT:
            return "*off0 = (*off1 > 0.0) ? floor(*off1 + 0.5) : ceil(*off1 - 0.5)";
        case BH_ISNAN:
            return "*off0 = isnan(*off1)";
        case BH_ISINF:
            return "*off0 = isinf(*off1)";
        case BH_IDENTITY:
            return "*off0 = *off1";

        default:
            return "__UNKNOWN__";

    }

}

