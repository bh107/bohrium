/*
 * Copyright 2011 Troels Blum <troels@blum.dk>
 *
 * This file is part of cphVB <http://code.google.com/p/cphvb/>.
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

#ifndef __CPHVB_OPCODE_H
#define __CPHVB_OPCODE_H

#include "cphvb_type.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Codes for known oparations */
enum /* cphvb_opcode */
{
    CPHVB_ADD,
    CPHVB_SUBTRACT,
    CPHVB_MULTIPLY,
    CPHVB_DIVIDE,
    CPHVB_LOGADDEXP,
    CPHVB_LOGADDEXP2,
    CPHVB_TRUE_DIVIDE,
    CPHVB_FLOOR_DIVIDE,
    CPHVB_NEGATIVE,
    CPHVB_POWER,
    CPHVB_REMAINDER,
    CPHVB_MOD,
    CPHVB_FMOD,
    CPHVB_ABSOLUTE,
    CPHVB_RINT,
    CPHVB_SIGN,
    CPHVB_CONJ,
    CPHVB_EXP,
    CPHVB_EXP2,
    CPHVB_LOG,
    CPHVB_LOG10,
    CPHVB_EXPM1,
    CPHVB_LOG1P,
    CPHVB_SQRT,
    CPHVB_SQUARE,
    CPHVB_RECIPROCAL,
    CPHVB_ONES_LIKE,
    CPHVB_SIN,
    CPHVB_COS,
    CPHVB_TAN,
    CPHVB_ARCSIN,
    CPHVB_ARCCOS,
    CPHVB_ARCTAN,
    CPHVB_ARCTAN2,
    CPHVB_HYPOT,
    CPHVB_SINH,
    CPHVB_COSH,
    CPHVB_TANH,
    CPHVB_ARCSINH,
    CPHVB_ARCCOSH,
    CPHVB_ARCTANH,
    CPHVB_DEG2RAD,
    CPHVB_RAD2DEG,
    CPHVB_BITWISE_AND,
    CPHVB_BITWISE_OR,
    CPHVB_BITWISE_XOR,
    CPHVB_LOGICAL_NOT,
    CPHVB_LOGICAL_AND,
    CPHVB_LOGICAL_OR,
    CPHVB_LOGICAL_XOR,
    CPHVB_INVERT,
    CPHVB_LEFT_SHIFT,
    CPHVB_RIGHT_SHIFT,
    CPHVB_GREATER,
    CPHVB_GREATER_EQUAL,
    CPHVB_LESS,
    CPHVB_LESS_EQUAL,
    CPHVB_NOT_EQUAL,
    CPHVB_EQUAL,
    CPHVB_MAXIMUM,
    CPHVB_MINIMUM,
    CPHVB_ISFINITE,
    CPHVB_ISINF,
    CPHVB_ISNAN,
    CPHVB_SIGNBIT,
    CPHVB_MODF,
    CPHVB_LDEXP,
    CPHVB_FREXP,
    CPHVB_FLOOR,
    CPHVB_CEIL,
    CPHVB_TRUNC,
    CPHVB_LOG2,
    CPHVB_ISREAL,
    CPHVB_ISCOMPLEX,
    CPHVB_RELEASE, // ==     CPHVB_SYNC + CPHVB_DISCARD
    CPHVB_SYNC,    //Inform child to make data synchronized and available.
    CPHVB_DISCARD, //Inform child to forget the array
    CPHVB_DESTROY, //Inform child to deallocate the array.
    CPHVB_RANDOM,  //file out with random
    CPHVB_ARANGE, // out, start, step
    //Used by a brigde to mark untranslatable operations.
    //NB: CPHVB_NONE must be the last element in this enum.
    CPHVB_NONE
};

#define CPHVB_NO_OPCODES CPHVB_NONE

#define CPHVB_REDUCE (1<<15)

#ifdef __cplusplus
}
#endif

#endif
