#include <fstream>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
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

const char* bhtypestr_to_shorthand(const char* type_str)
{
    if (strcmp("BH_BOOL", type_str)==0) {
        return "z";
    } else if (strcmp("BH_INT8", type_str)==0) {
        return "b";
    } else if (strcmp("BH_INT16", type_str)==0) {
        return "s";
    } else if (strcmp(type_str, "BH_INT32")==0) {
        return "i";
    } else if (strcmp(type_str, "BH_INT64")==0) {
        return "l";
    } else if (strcmp(type_str, "BH_UINT8")==0) {
        return "B";
    } else if (strcmp(type_str, "BH_UINT16")==0) {
        return "S";
    } else if (strcmp(type_str, "BH_UINT32")==0) {
        return "I";
    } else if (strcmp(type_str, "BH_UINT64")==0) {
        return "L";
    } else if (strcmp(type_str, "BH_FLOAT16")==0) {
        return "h";
    } else if (strcmp(type_str, "BH_FLOAT32")==0) {
        return "f";
    } else if (strcmp(type_str, "BH_FLOAT64")==0) {
        return "d";
    } else if (strcmp(type_str, "BH_COMPLEX64")==0) {
        return "c";
    } else if (strcmp(type_str, "BH_COMPLEX128")==0) {
        return "C";
    } else {
        return "UNKNOWN";
    }
}

const char* typestr_to_ctype(const char* type_str)
{
    if (strcmp("BH_BOOL", type_str)==0) {
        return "unsigned char";
    } else if (strcmp("BH_INT8", type_str)==0) {
        return "int8_t";
    } else if (strcmp("BH_INT16", type_str)==0) {
        return "int16_t";
    } else if (strcmp(type_str, "BH_INT32")==0) {
        return "int32_t";
    } else if (strcmp(type_str, "BH_INT64")==0) {
        return "int64_t";
    } else if (strcmp(type_str, "BH_UINT8")==0) {
        return "uint8_t";
    } else if (strcmp(type_str, "BH_UINT16")==0) {
        return "uint16_t";
    } else if (strcmp(type_str, "BH_UINT32")==0) {
        return "uint32_t";
    } else if (strcmp(type_str, "BH_UINT64")==0) {
        return "uint64_t";
    } else if (strcmp(type_str, "BH_FLOAT16")==0) {
        return "uint16_t";
    } else if (strcmp(type_str, "BH_FLOAT32")==0) {
        return "float";
    } else if (strcmp(type_str, "BH_FLOAT64")==0) {
        return "double";
    } else if (strcmp(type_str, "BH_COMPLEX64")==0) {
        return "complex float";
    } else if (strcmp(type_str, "BH_COMPLEX128")==0) {
        return "complex double";
    } else {
        return "UNKNOWN";
    }
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

const char* cexpr(bh_opcode opcode)
{
     switch(opcode) {
        case BH_ADD_REDUCE:
            return "%s += %s";

        // Binary elementwise: ADD, MULTIPLY...
        case BH_ADD:
            return "%s + %s";
        case BH_SUBTRACT:
            return "%s - %s";
        case BH_MULTIPLY:
            return "%s * %s";
        case BH_DIVIDE:
            return "%s / %s";
        case BH_POWER:
            return "pow(%s, %s)";
        case BH_LESS_EQUAL:
            return "%s <= %s";
        case BH_EQUAL:
            return "%s == %s";

        case BH_SQRT:
            return "sqrt(%s)";
        case BH_IDENTITY:
            return "%s";

        default:
            return "__UNKNOWN__";
    }   
}

const char* bhopcode_to_cexpr(bh_opcode opcode)
{
    switch(opcode) {
        /*
        case BH_ADD_REDUCE:
            return "*a0_current += *tmp_current";
        case BH_MULTIPLY_REDUCE:
            return "*a0_current *= *tmp_current";
        case BH_MINIMUM_REDUCE:
            return "*a0_current = *a0_current < *tmp_current ? *a0_current : *tmp_current";
        case BH_MAXIMUM_REDUCE:
            return "*a0_current = *a0_current < *tmp_current ? *tmp_current : *a0_current";
        case BH_LOGICAL_AND_REDUCE:
            return "*a0_current = *a0_current && *tmp_current";
        case BH_BITWISE_AND_REDUCE:
            return "*a0_current &= *tmp_current";
        case BH_LOGICAL_OR_REDUCE:
            return "*a0_current = *a0_current || *tmp_current";
        case BH_BITWISE_OR_REDUCE:
            return "*a0_current |= *tmp_current";

        case BH_LOGICAL_XOR_REDUCE:
            return "*a0_current = !*a0_current != !*tmp_current";
        case BH_BITWISE_XOR_REDUCE:
            return "*a0_current = *a0_current ^ *tmp_current";
        */

        case BH_ADD_REDUCE:
            return "rvar += *tmp_current";
        case BH_MULTIPLY_REDUCE:
            return "rvar *= *tmp_current";
        case BH_MINIMUM_REDUCE:
            return "rvar = rvar < *tmp_current ? rvar : *tmp_current";
        case BH_MAXIMUM_REDUCE:
            return "rvar = rvar < *tmp_current ? *tmp_current : rvar";
        case BH_LOGICAL_AND_REDUCE:
            return "rvar = rvar && *tmp_current";
        case BH_BITWISE_AND_REDUCE:
            return "rvar &= *tmp_current";
        case BH_LOGICAL_OR_REDUCE:
            return "rvar = rvar || *tmp_current";
        case BH_BITWISE_OR_REDUCE:
            return "rvar |= *tmp_current";

        case BH_LOGICAL_XOR_REDUCE:
            return "rvar = !rvar != !*tmp_current";
        case BH_BITWISE_XOR_REDUCE:
            return "rvar = rvar ^ *tmp_current";

        // Binary elementwise: ADD, MULTIPLY...
        case BH_ADD:
            return "*a0_current = *a1_current + *a2_current";
        case BH_SUBTRACT:
            return "*a0_current = *a1_current - *a2_current";
        case BH_MULTIPLY:
            return "*a0_current = *a1_current * *a2_current";
        case BH_DIVIDE:
            return "*a0_current = *a1_current / *a2_current";
        case BH_POWER:
            return "*a0_current = pow( *a1_current, *a2_current )";
        case BH_GREATER:
            return "*a0_current = *a1_current > *a2_current";
        case BH_GREATER_EQUAL:
            return "*a0_current = *a1_current >= *a2_current";
        case BH_LESS:
            return "*a0_current = *a1_current < *a2_current";
        case BH_LESS_EQUAL:
            return "*a0_current = *a1_current <= *a2_current";
        case BH_EQUAL:
            return "*a0_current = *a1_current == *a2_current";
        case BH_NOT_EQUAL:
            return "*a0_current = *a1_current != *a2_current";
        case BH_LOGICAL_AND:
            return "*a0_current = *a1_current && *a2_current";
        case BH_LOGICAL_OR:
            return "*a0_current = *a1_current || *a2_current";
        case BH_LOGICAL_XOR:
            return "*a0_current = (!*a1_current != !*a2_current)";
        case BH_MAXIMUM:
            return "*a0_current = *a1_current < *a2_current ? *a2_current : *a1_current";
        case BH_MINIMUM:
            return "*a0_current = *a1_current < *a2_current ? *a1_current : *a2_current";
        case BH_BITWISE_AND:
            return "*a0_current = *a1_current & *a2_current";
        case BH_BITWISE_OR:
            return "*a0_current = *a1_current | *a2_current";
        case BH_BITWISE_XOR:
            return "*a0_current = *a1_current ^ *a2_current";
        case BH_LEFT_SHIFT:
            return "*a0_current = (*a1_current) << (*a2_current)";
        case BH_RIGHT_SHIFT:
            return "*a0_current = (*a1_current) >> (*a2_current)";
        case BH_ARCTAN2:
            return "*a0_current = atan2( *a1_current, *a2_current )";
        case BH_MOD:
            return "*a0_current = *a1_current - floor(*a1_current / *a2_current) * *a2_current";

        // Unary elementwise: SQRT, SIN...
        case BH_ABSOLUTE:
            return "*a0_current = *a1_current < 0.0 ? -*a1_current: *a1_current";
        case BH_LOGICAL_NOT:
            return "*a0_current = !*a1_current";
        case BH_INVERT:
            return "*a0_current = ~*a1_current";
        case BH_COS:
            return "*a0_current = cos( *a1_current )";
        case BH_SIN:
            return "*a0_current = sin( *a1_current )";
        case BH_TAN:
            return "*a0_current = tan( *a1_current )";
        case BH_COSH:
            return "*a0_current = cosh( *a1_current )";
        case BH_SINH:
            return "*a0_current = sinh( *a1_current )";
        case BH_TANH:
            return "*a0_current = tanh( *a1_current )";
        case BH_ARCSIN:
            return "*a0_current = asin( *a1_current )";
        case BH_ARCCOS:
            return "*a0_current = acos( *a1_current )";
        case BH_ARCTAN:
            return "*a0_current = atan( *a1_current )";
        case BH_ARCSINH:
            return "*a0_current = asinh( *a1_current )";
        case BH_ARCCOSH:
            return "*a0_current = acosh( *a1_current )";
        case BH_ARCTANH:
            return "*a0_current = atanh( *a1_current )";
        case BH_EXP:
            return "*a0_current = exp( *a1_current )";
        case BH_EXP2:
            return "*a0_current = pow( 2, *a1_current )";
        case BH_EXPM1:
            return "*a0_current = expm1( *a1_current )";
        case BH_LOG:
            return "*a0_current = log( *a1_current )";
        case BH_LOG2:
            return "*a0_current = log2( *a1_current )";
        case BH_LOG10:
            return "*a0_current = log10( *a1_current )";
        case BH_LOG1P:
            return "*a0_current = log1p( *a1_current )";
        case BH_SQRT:
            return "*a0_current = sqrt( *a1_current )";
        case BH_CEIL:
            return "*a0_current = ceil( *a1_current )";
        case BH_TRUNC:
            return "*a0_current = trunc( *a1_current )";
        case BH_FLOOR:
            return "*a0_current = floor( *a1_current )";
        case BH_RINT:
            return "*a0_current = (*a1_current > 0.0) ? floor(*a1_current + 0.5) : ceil(*a1_current - 0.5)";
        case BH_ISNAN:
            return "*a0_current = isnan(*a1_current)";
        case BH_ISINF:
            return "*a0_current = isinf(*a1_current)";
        case BH_IDENTITY:
            return "*a0_current = *a1_current";

        default:
            return "__UNKNOWN__";
    }
}


std::string const_as_string(bh_constant constant)
{
    std::ostringstream buff;
    switch(constant.type) {
        case BH_BOOL:
            buff << constant.value.bool8;
            break;
        case BH_INT8:
            buff << constant.value.int8;
            break;
        case BH_INT16:
            buff << constant.value.int16;
            break;
        case BH_INT32:
            buff << constant.value.int32;
            break;
        case BH_INT64:
            buff << constant.value.int64;
            break;
        case BH_UINT8:
            buff << constant.value.uint8;
            break;
        case BH_UINT16:
            buff << constant.value.uint16;
            break;
        case BH_UINT32:
            buff << constant.value.uint32;
            break;
        case BH_UINT64:
            buff << constant.value.uint64;
            break;
        case BH_FLOAT16:
            buff << constant.value.float16;
            break;
        case BH_FLOAT32:
            buff << constant.value.float32;
            break;
        case BH_FLOAT64:
            buff << constant.value.float64;
            break;
        case BH_COMPLEX64:
            buff << constant.value.complex64.real << constant.value.complex64.imag;
            break;
        case BH_COMPLEX128:
            buff << constant.value.complex128.real << constant.value.complex128.imag;
            break;

        case BH_UNKNOWN:
        default:
            buff << "__ERROR__";
    }

    return buff.str();
}


