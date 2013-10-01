#include <fstream>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include "bh.h"
#include "bh_ve_cpu.h"

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

inline
bool is_dense(bh_view *operand)
{
    if ((operand->ndim == 3) && \
        (operand->stride[0] == 1) && \
        (operand->stride[1] == operand->shape[0]) && \
        (operand->stride[2] == operand->shape[1]*operand->shape[0])
    ) {
        return true;
    } else if ((operand->ndim == 2) && \
               (operand->stride[0] == 1) && \
               (operand->stride[1] == operand->shape[0])) {
        return true;
    } else if ((operand->ndim == 1) && (operand->stride[0] == 1)) {
        return true;
    }

    return false;
}

int bh_layoutmask(bh_instruction *instr)
{
    int mask = 0;
    const int nops = bh_operands(instr->opcode);

    switch(nops) {
        case 3:
            mask |= (is_dense(&instr->operand[0])) ? A0_DENSE : A0_STRIDED;
            if (bh_is_constant(&instr->operand[2])) {
                mask |= (is_dense(&instr->operand[1])) ? A1_DENSE : A1_STRIDED;
                mask |= A2_CONSTANT;
            } else if (bh_is_constant(&instr->operand[1])) {
                mask |= A1_CONSTANT;
                mask |= (is_dense(&instr->operand[2])) ? A2_DENSE : A2_STRIDED;
            } else {
                mask |= (is_dense(&instr->operand[1])) ? A1_DENSE : A1_STRIDED;
                mask |= (is_dense(&instr->operand[2])) ? A2_DENSE : A2_STRIDED;
            }
            break;

        case 2:
            mask |= (is_dense(&instr->operand[0])) ? A0_DENSE : A0_STRIDED;
            if (bh_is_constant(&instr->operand[1])) {
                mask |= A1_CONSTANT;
            } else {
                mask |= (is_dense(&instr->operand[1])) ? A1_DENSE : A1_STRIDED;
            }
            break;

        case 1:
            mask |= (is_dense(&instr->operand[0])) ? A0_DENSE : A0_STRIDED;
            if (bh_is_constant(&instr->operand[1])) {
                mask |= A1_CONSTANT;
            } else {
                mask |= (is_dense(&instr->operand[1])) ? A1_DENSE : A1_STRIDED;
            }           
            break;

        case 0:
        default:
            break;
    }
    return mask;
}

const char* bh_layoutmask_to_shorthand(const int mask)
{
    switch(mask) {
        case 1: return "C";
        case 2: return "D";
        case 4: return "S";
        case 8: return "P";

        case 17: return "CC";
        case 18: return "DC";
        case 20: return "SC";
        case 24: return "PC";
        case 33: return "CD";
        case 34: return "DD";
        case 36: return "SD";
        case 40: return "PD";
        case 65: return "CS";
        case 66: return "DS";
        case 68: return "SS";
        case 72: return "PS";
        case 129: return "CP";
        case 130: return "DP";
        case 132: return "SP";
        case 136: return "PP";
        case 273: return "CCC";
        case 274: return "DCC";
        case 276: return "SCC";
        case 280: return "PCC";
        case 289: return "CDC";
        case 290: return "DDC";
        case 292: return "SDC";
        case 296: return "PDC";
        case 321: return "CSC";
        case 322: return "DSC";
        case 324: return "SSC";
        case 328: return "PSC";
        case 385: return "CPC";
        case 386: return "DPC";
        case 388: return "SPC";
        case 392: return "PPC";
        case 529: return "CCD";
        case 530: return "DCD";
        case 532: return "SCD";
        case 536: return "PCD";
        case 545: return "CDD";
        case 546: return "DDD";
        case 548: return "SDD";
        case 552: return "PDD";
        case 577: return "CSD";
        case 578: return "DSD";
        case 580: return "SSD";
        case 584: return "PSD";
        case 641: return "CPD";
        case 642: return "DPD";
        case 644: return "SPD";
        case 648: return "PPD";
        case 1041: return "CCS";
        case 1042: return "DCS";
        case 1044: return "SCS";
        case 1048: return "PCS";
        case 1057: return "CDS";
        case 1058: return "DDS";
        case 1060: return "SDS";
        case 1064: return "PDS";
        case 1089: return "CSS";
        case 1090: return "DSS";
        case 1092: return "SSS";
        case 1096: return "PSS";
        case 1153: return "CPS";
        case 1154: return "DPS";
        case 1156: return "SPS";
        case 1160: return "PPS";
        case 2065: return "CCP";
        case 2066: return "DCP";
        case 2068: return "SCP";
        case 2072: return "PCP";
        case 2081: return "CDP";
        case 2082: return "DDP";
        case 2084: return "SDP";
        case 2088: return "PDP";
        case 2113: return "CSP";
        case 2114: return "DSP";
        case 2116: return "SSP";
        case 2120: return "PSP";
        case 2177: return "CPP";
        case 2178: return "DPP";
        case 2180: return "SPP";
        case 2184: return "PPP";

        default:
            return "___";
    }
}

int bh_typesig(bh_instruction *instr)
{
    int typesig;
    const int nops = bh_operands(instr->opcode);
    switch(nops) {
        case 3:
            typesig = instr->operand[0].base->type;

            if (bh_is_constant(&instr->operand[1])) {             
                typesig += ((1+instr->constant.type) << 4) \
                          +((1+instr->operand[2].base->type) << 8);

            } else if (bh_is_constant(&instr->operand[2])) {      
                typesig = ((1+instr->operand[1].base->type) << 4) \
                         +((1+instr->constant.type) << 8);

            } else {                                                
                typesig = ((1+instr->operand[1].base->type) << 4) \
                         +((1+instr->operand[2].base->type) << 8);
            }
            break;
        case 2:
            typesig = instr->operand[0].base->type;

            if (bh_is_constant(&instr->operand[1])) {
                typesig = ((1+instr->constant.type) << 4);
            } else {
                typesig = ((1+instr->operand[1].base->type) << 4);
            }
            break;
        case 1:
            typesig = (1+instr->operand[0].base->type);
            break;
        case 0:
        default:
            typesig = 0;
            break;
    }
    return typesig;
}

const char* bh_typesig_to_shorthand(int typesig)
{
    
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


