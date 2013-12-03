#include <fstream>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include "bh.h"
#include "bh_ve_cpu.h"

#include "utils.auto.cpp"

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

/**
 * Compute the layoutmask of the instruction.
 *
 */
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

int bh_typesig(bh_instruction *instr)
{
    int typesig;
    const int nops = bh_operands(instr->opcode);
    switch(nops) {
        case 3:
            typesig = instr->operand[0].base->type+1;

            if (bh_is_constant(&instr->operand[1])) {             
                typesig += ((1+instr->constant.type) << 4) \
                          +((1+instr->operand[2].base->type) << 8);

            } else if (bh_is_constant(&instr->operand[2])) {      
                typesig += ((1+instr->operand[1].base->type) << 4) \
                          +((1+instr->constant.type) << 8);

            } else {                                                
                typesig += ((1+instr->operand[1].base->type) << 4) \
                          +((1+instr->operand[2].base->type) << 8);
            }
            break;
        case 2:
            typesig = instr->operand[0].base->type+1;

            if (bh_is_constant(&instr->operand[1])) {
                typesig += ((1+instr->constant.type) << 4);
            } else {
                typesig += ((1+instr->operand[1].base->type) << 4);
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
        case BH_RANDOM:
            return  "threefry2x64_ctr_t ctr = {{*a2_current, 0}};            //index\n" \
                    "threefry2x64_key_t key = {{*a1_current, 0xdeadbeef}};  //seed\n"  \
                    "threefry2x64_ctr_t   c = threefry2x64(ctr, key);\n"       \
                    "*a0_current = c.v[0];\n";

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

