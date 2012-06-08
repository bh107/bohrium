/*
 * Copyright 2011 Troels Blum <troels@blum.dk>
 *
 * This file is part of cphVB.
 *
 * cphVB is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * cphVB is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with cphVB. If not, see <http://www.gnu.org/licenses/>.
 */

#include <stdlib.h>
#include <stdio.h>
#include <cphvb_opcode.h>
#include <cphvb.h>
#include <stdbool.h>


/* Number of operands for operation
 *
 * @opcode Opcode for operation
 * @return Number of operands
 */
int cphvb_operands(cphvb_opcode opcode)
{
    // The reduce family of operations take the same number of arguments
    // as the regular operation
    
    switch(opcode) 
    {
    case CPHVB_ADD: return 3;
    case CPHVB_SUBTRACT: return 3;
    case CPHVB_MULTIPLY: return 3;
    case CPHVB_DIVIDE: return 3;
    case CPHVB_LOGADDEXP: return 3;
    case CPHVB_LOGADDEXP2: return 3;
    case CPHVB_TRUE_DIVIDE: return 3;
    case CPHVB_FLOOR_DIVIDE: return 3;
    case CPHVB_NEGATIVE: return 2;
    case CPHVB_POWER: return 3;
    case CPHVB_REMAINDER: return 3;
    case CPHVB_MOD: return 3;
    case CPHVB_FMOD: return 3;
    case CPHVB_ABSOLUTE: return 2;
    case CPHVB_RINT: return 2;
    case CPHVB_SIGN: return 2;
    case CPHVB_CONJ: return 2;
    case CPHVB_EXP: return 2;
    case CPHVB_EXP2: return 2;
    case CPHVB_LOG: return 2;
    case CPHVB_LOG10: return 2;
    case CPHVB_EXPM1: return 2;
    case CPHVB_LOG1P: return 2;
    case CPHVB_SQRT: return 2;
    case CPHVB_SQUARE: return 2;
    case CPHVB_RECIPROCAL: return 2;
    case CPHVB_ONES_LIKE: return 2;
    case CPHVB_SIN: return 2;
    case CPHVB_COS: return 2;
    case CPHVB_TAN: return 2;
    case CPHVB_ARCSIN: return 2;
    case CPHVB_ARCCOS: return 2;
    case CPHVB_ARCTAN: return 2;
    case CPHVB_ARCTAN2: return 3;
    case CPHVB_HYPOT: return 3;
    case CPHVB_SINH: return 2;
    case CPHVB_COSH: return 2;
    case CPHVB_TANH: return 2;
    case CPHVB_ARCSINH: return 2;
    case CPHVB_ARCCOSH: return 2;
    case CPHVB_ARCTANH: return 2;
    case CPHVB_DEG2RAD: return 2;
    case CPHVB_RAD2DEG: return 2;
    case CPHVB_BITWISE_AND: return 3;
    case CPHVB_BITWISE_OR: return 3;
    case CPHVB_BITWISE_XOR: return 3;
    case CPHVB_LOGICAL_NOT: return 2;
    case CPHVB_LOGICAL_AND: return 3;
    case CPHVB_LOGICAL_OR: return 3;
    case CPHVB_LOGICAL_XOR: return 3;
    case CPHVB_INVERT: return 2;
    case CPHVB_LEFT_SHIFT: return 3;
    case CPHVB_RIGHT_SHIFT: return 3;
    case CPHVB_GREATER: return 3;
    case CPHVB_GREATER_EQUAL: return 3;
    case CPHVB_LESS: return 3;
    case CPHVB_LESS_EQUAL: return 3;
    case CPHVB_NOT_EQUAL: return 3;
    case CPHVB_EQUAL: return 3;
    case CPHVB_MAXIMUM: return 3;
    case CPHVB_MINIMUM: return 3;
    case CPHVB_ISFINITE: return 2;
    case CPHVB_ISINF: return 2;
    case CPHVB_ISNAN: return 2;
    case CPHVB_SIGNBIT: return 2;
    case CPHVB_MODF: return 3;
    case CPHVB_LDEXP: return 3;
    case CPHVB_FREXP: return 3;
    case CPHVB_FLOOR: return 2;
    case CPHVB_CEIL: return 2;
    case CPHVB_TRUNC: return 2;
    case CPHVB_LOG2: return 2;
    case CPHVB_ISREAL: return 2;
    case CPHVB_ISCOMPLEX: return 2;
    case CPHVB_SYNC: return 1;
    case CPHVB_DISCARD: return 1;
    case CPHVB_DESTROY: return 1;
    case CPHVB_RANDOM: return 1;
    case CPHVB_ARANGE: return 3;
    case CPHVB_IDENTITY: return 2;
    case CPHVB_USERFUNC: return 0;
    case CPHVB_NONE: return 0; 	
    default: return -1;
    }
}

/* Number of operands in instruction
 * NB: this function handles user-defined function correctly
 * @inst Instruction
 * @return Number of operands
 */
int cphvb_operands_in_instruction(cphvb_instruction *inst)
{
    if (inst->opcode == CPHVB_USERFUNC)
        return inst->userfunc->nin + inst->userfunc->nout;
    else
        return cphvb_operands(inst->opcode);
}


/* Text descriptions for a given operation */
const char* _opcode_text[CPHVB_NONE+1];
bool _opcode_text_initialized = false;

/* Text string for operation
 *
 * @opcode Opcode for operation
 * @return Text string.
 */
const char* cphvb_opcode_text(cphvb_opcode opcode)
{
    switch(opcode)
    {
    case CPHVB_ADD: return "CPHVB_ADD";
    case CPHVB_SUBTRACT: return "CPHVB_SUBTRACT";
    case CPHVB_MULTIPLY: return "CPHVB_MULTIPLY";
    case CPHVB_DIVIDE: return "CPHVB_DIVIDE";
    case CPHVB_LOGADDEXP: return "CPHVB_LOGADDEXP";
    case CPHVB_LOGADDEXP2: return "CPHVB_LOGADDEXP2";
    case CPHVB_TRUE_DIVIDE: return "CPHVB_TRUE_DIVIDE";
    case CPHVB_FLOOR_DIVIDE: return "CPHVB_FLOOR_DIVIDE";
    case CPHVB_NEGATIVE: return "CPHVB_NEGATIVE";
    case CPHVB_POWER: return "CPHVB_POWER";
    case CPHVB_REMAINDER: return "CPHVB_REMAINDER";
    case CPHVB_MOD: return "CPHVB_MOD";
    case CPHVB_FMOD: return "CPHVB_FMOD";
    case CPHVB_ABSOLUTE: return "CPHVB_ABSOLUTE";
    case CPHVB_RINT: return "CPHVB_RINT";
    case CPHVB_SIGN: return "CPHVB_SIGN";
    case CPHVB_CONJ: return "CPHVB_CONJ";
    case CPHVB_EXP: return "CPHVB_EXP";
    case CPHVB_EXP2: return "CPHVB_EXP2";
    case CPHVB_LOG: return "CPHVB_LOG";
    case CPHVB_LOG10: return "CPHVB_LOG10";
    case CPHVB_EXPM1: return "CPHVB_EXPM1";
    case CPHVB_LOG1P: return "CPHVB_LOG1P";
    case CPHVB_SQRT: return "CPHVB_SQRT";
    case CPHVB_SQUARE: return "CPHVB_SQUARE";
    case CPHVB_RECIPROCAL: return "CPHVB_RECIPROCAL";
    case CPHVB_ONES_LIKE: return "CPHVB_ONES_LIKE";
    case CPHVB_SIN: return "CPHVB_SIN";
    case CPHVB_COS: return "CPHVB_COS";
    case CPHVB_TAN: return "CPHVB_TAN";
    case CPHVB_ARCSIN: return "CPHVB_ARCSIN";
    case CPHVB_ARCCOS: return "CPHVB_ARCCOS";
    case CPHVB_ARCTAN: return "CPHVB_ARCTAN";
    case CPHVB_ARCTAN2: return "CPHVB_ARCTAN2";
    case CPHVB_HYPOT: return "CPHVB_HYPOT";
    case CPHVB_SINH: return "CPHVB_SINH";
    case CPHVB_COSH: return "CPHVB_COSH";
    case CPHVB_TANH: return "CPHVB_TANH";
    case CPHVB_ARCSINH: return "CPHVB_ARCSINH";
    case CPHVB_ARCCOSH: return "CPHVB_ARCCOSH";
    case CPHVB_ARCTANH: return "CPHVB_ARCTANH";
    case CPHVB_DEG2RAD: return "CPHVB_DEG2RAD";
    case CPHVB_RAD2DEG: return "CPHVB_RAD2DEG";
    case CPHVB_BITWISE_AND: return "CPHVB_BITWISE_AND";
    case CPHVB_BITWISE_OR: return "CPHVB_BITWISE_OR";
    case CPHVB_BITWISE_XOR: return "CPHVB_BITWISE_XOR";
    case CPHVB_LOGICAL_NOT: return "CPHVB_LOGICAL_NOT";
    case CPHVB_LOGICAL_AND: return "CPHVB_LOGICAL_AND";
    case CPHVB_LOGICAL_OR: return "CPHVB_LOGICAL_OR";
    case CPHVB_LOGICAL_XOR: return "CPHVB_LOGICAL_XOR";
    case CPHVB_INVERT: return "CPHVB_INVERT";
    case CPHVB_LEFT_SHIFT: return "CPHVB_LEFT_SHIFT";
    case CPHVB_RIGHT_SHIFT: return "CPHVB_RIGHT_SHIFT";
    case CPHVB_GREATER: return "CPHVB_GREATER";
    case CPHVB_GREATER_EQUAL: return "CPHVB_GREATER_EQUAL";
    case CPHVB_LESS: return "CPHVB_LESS";
    case CPHVB_LESS_EQUAL: return "CPHVB_LESS_EQUAL";
    case CPHVB_NOT_EQUAL: return "CPHVB_NOT_EQUAL";
    case CPHVB_EQUAL: return "CPHVB_EQUAL";
    case CPHVB_MAXIMUM: return "CPHVB_MAXIMUM";
    case CPHVB_MINIMUM: return "CPHVB_MINIMUM";
    case CPHVB_ISFINITE: return "CPHVB_ISFINITE";
    case CPHVB_ISINF: return "CPHVB_ISINF";
    case CPHVB_ISNAN: return "CPHVB_ISNAN";
    case CPHVB_SIGNBIT: return "CPHVB_SIGNBIT";
    case CPHVB_MODF: return "CPHVB_MODF";
    case CPHVB_LDEXP: return "CPHVB_LDEXP";
    case CPHVB_FREXP: return "CPHVB_FREXP";
    case CPHVB_FLOOR: return "CPHVB_FLOOR";
    case CPHVB_CEIL: return "CPHVB_CEIL";
    case CPHVB_TRUNC: return "CPHVB_TRUNC";
    case CPHVB_LOG2: return "CPHVB_LOG2";
    case CPHVB_ISREAL: return "CPHVB_ISREAL";
    case CPHVB_ISCOMPLEX: return "CPHVB_ISCOMPLEX";
    case CPHVB_SYNC: return "CPHVB_SYNC";
    case CPHVB_DISCARD: return "CPHVB_DISCARD";
    case CPHVB_DESTROY: return "CPHVB_DESTROY";
    case CPHVB_RANDOM: return "CPHVB_RANDOM";
    case CPHVB_ARANGE: return "CPHVB_ARANGE";
    case CPHVB_IDENTITY: return "CPHVB_IDENTITY";
    case CPHVB_USERFUNC: return "CPHVB_USERFUNC";
    case CPHVB_NONE: return "CPHVB_NONE";
    default: return "Unknown opcode";
    }
}
