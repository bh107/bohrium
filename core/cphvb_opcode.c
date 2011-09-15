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

/* Number of operands for a given operation */
const int _operands[] =
{
    [CPHVB_ADD] = 3,
    [CPHVB_SUBTRACT] = 3,
    [CPHVB_MULTIPLY] = 3,
    [CPHVB_DIVIDE] = 3,
    [CPHVB_LOGADDEXP] = 3,
    [CPHVB_LOGADDEXP2] = 3,
    [CPHVB_TRUE_DIVIDE] = 3,
    [CPHVB_FLOOR_DIVIDE] = 3,
    [CPHVB_NEGATIVE] = 2,
    [CPHVB_POWER] = 3,
    [CPHVB_REMAINDER] = 3,
    [CPHVB_MOD] = 3,
    [CPHVB_FMOD] = 3,
    [CPHVB_ABSOLUTE] = 2,
    [CPHVB_RINT] = 2,
    [CPHVB_SIGN] = 2,
    [CPHVB_CONJ] = 2,
    [CPHVB_EXP] = 2,
    [CPHVB_EXP2] = 2,
    [CPHVB_LOG] = 2,
    [CPHVB_LOG10] = 2,
    [CPHVB_EXPM1] = 2,
    [CPHVB_LOG1P] = 2,
    [CPHVB_SQRT] = 2,
    [CPHVB_SQUARE] = 2,
    [CPHVB_RECIPROCAL] = 2,
    [CPHVB_ONES_LIKE] = 2,
    [CPHVB_SIN] = 2,
    [CPHVB_COS] = 2,
    [CPHVB_TAN] = 2,
    [CPHVB_ARCSIN] = 2,
    [CPHVB_ARCCOS] = 2,
    [CPHVB_ARCTAN] = 2,
    [CPHVB_ARCTAN2] = 3,
    [CPHVB_HYPOT] = 3,
    [CPHVB_SINH] = 2,
    [CPHVB_COSH] = 2,
    [CPHVB_TANH] = 2,
    [CPHVB_ARCSINH] = 2,
    [CPHVB_ARCCOSH] = 2,
    [CPHVB_ARCTANH] = 2,
    [CPHVB_DEG2RAD] = 2,
    [CPHVB_RAD2DEG] = 2,
    [CPHVB_BITWISE_AND] = 3,
    [CPHVB_BITWISE_OR] = 3,
    [CPHVB_BITWISE_XOR] = 3,
    [CPHVB_LOGICAL_NOT] = 3,
    [CPHVB_LOGICAL_AND] = 3,
    [CPHVB_LOGICAL_OR] = 3,
    [CPHVB_LOGICAL_XOR] = 3,
    [CPHVB_INVERT] = 2,
    [CPHVB_LEFT_SHIFT] = 3,
    [CPHVB_RIGHT_SHIFT] = 3,
    [CPHVB_GREATER] = 3,
    [CPHVB_GREATER_EQUAL] = 3,
    [CPHVB_LESS] = 3,
    [CPHVB_LESS_EQUAL] = 3,
    [CPHVB_NOT_EQUAL] = 3,
    [CPHVB_EQUAL] = 3,
    [CPHVB_MAXIMUM] = 3,
    [CPHVB_MINIMUM] = 3,
    [CPHVB_ISFINITE] = 2,
    [CPHVB_ISINF] = 2,
    [CPHVB_ISNAN] = 2,
    [CPHVB_SIGNBIT] = 2,
    [CPHVB_MODF] = 3,
    [CPHVB_LDEXP] = 3,
    [CPHVB_FREXP] = 3,
    [CPHVB_FLOOR] = 2,
    [CPHVB_CEIL] = 2,
    [CPHVB_TRUNC] = 2,
    [CPHVB_LOG2] = 2,
    [CPHVB_ISREAL] = 2,
    [CPHVB_ISCOMPLEX] = 2,
    [CPHVB_RELEASE] = 1,
    [CPHVB_SYNC] = 1,
    [CPHVB_DISCARD] = 1,
    [CPHVB_DESTROY] = 1,
    [CPHVB_RANDOM] = 1,
    [CPHVB_ARANGE] = 3,
    [CPHVB_NONE] = 0
};


/* Number of operands for operation
 *
 * @opcode Opcode for operation
 * @return Number of operands
 */
int cphvb_operands(cphvb_opcode opcode)
{
    // The reduce family of operations take the same number of arguments 
    // as the regular operation
    return _operands[~CPHVB_REDUCE & opcode];
}

const char* const _opcode_text[] =
{
    [CPHVB_ADD] = "CPHVB_ADD",
    [CPHVB_SUBTRACT] = "CPHVB_SUBTRACT",
    [CPHVB_MULTIPLY] = "CPHVB_MULTIPLY",
    [CPHVB_DIVIDE] = "CPHVB_DIVIDE",
    [CPHVB_LOGADDEXP] = "CPHVB_LOGADDEXP",
    [CPHVB_LOGADDEXP2] = "CPHVB_LOGADDEXP2",
    [CPHVB_TRUE_DIVIDE] = "CPHVB_TRUE_DIVIDE",
    [CPHVB_FLOOR_DIVIDE] = "CPHVB_FLOOR_DIVIDE",
    [CPHVB_NEGATIVE] = "CPHVB_NEGATIVE",
    [CPHVB_POWER] = "CPHVB_POWER",
    [CPHVB_REMAINDER] = "CPHVB_REMAINDER",
    [CPHVB_MOD] = "CPHVB_MOD",
    [CPHVB_FMOD] = "CPHVB_FMOD",
    [CPHVB_ABSOLUTE] = "CPHVB_ABSOLUTE",
    [CPHVB_RINT] = "CPHVB_RINT",
    [CPHVB_SIGN] = "CPHVB_SIGN",
    [CPHVB_CONJ] = "CPHVB_CONJ",
    [CPHVB_EXP] = "CPHVB_EXP",
    [CPHVB_EXP2] = "CPHVB_EXP2",
    [CPHVB_LOG] = "CPHVB_LOG",
    [CPHVB_LOG10] = "CPHVB_LOG10",
    [CPHVB_EXPM1] = "CPHVB_EXPM1",
    [CPHVB_LOG1P] = "CPHVB_LOG1P",
    [CPHVB_SQRT] = "CPHVB_SQRT",
    [CPHVB_SQUARE] = "CPHVB_SQUARE",
    [CPHVB_RECIPROCAL] = "CPHVB_RECIPROCAL",
    [CPHVB_ONES_LIKE] = "CPHVB_ONES_LIKE",
    [CPHVB_SIN] = "CPHVB_SIN",
    [CPHVB_COS] = "CPHVB_COS",
    [CPHVB_TAN] = "CPHVB_TAN",
    [CPHVB_ARCSIN] = "CPHVB_ARCSIN",
    [CPHVB_ARCCOS] = "CPHVB_ARCCOS",
    [CPHVB_ARCTAN] = "CPHVB_ARCTAN",
    [CPHVB_ARCTAN2] = "CPHVB_ARCTAN2",
    [CPHVB_HYPOT] = "CPHVB_HYPOT",
    [CPHVB_SINH] = "CPHVB_SINH",
    [CPHVB_COSH] = "CPHVB_COSH",
    [CPHVB_TANH] = "CPHVB_TANH",
    [CPHVB_ARCSINH] = "CPHVB_ARCSINH",
    [CPHVB_ARCCOSH] = "CPHVB_ARCCOSH",
    [CPHVB_ARCTANH] = "CPHVB_ARCTANH",
    [CPHVB_DEG2RAD] = "CPHVB_DEG2RAD",
    [CPHVB_RAD2DEG] = "CPHVB_RAD2DEG",
    [CPHVB_BITWISE_AND] = "CPHVB_BITWISE_AND",
    [CPHVB_BITWISE_OR] = "CPHVB_BITWISE_OR",
    [CPHVB_BITWISE_XOR] = "CPHVB_BITWISE_XOR",
    [CPHVB_LOGICAL_NOT] = "CPHVB_LOGICAL_NOT",
    [CPHVB_LOGICAL_AND] = "CPHVB_LOGICAL_AND",
    [CPHVB_LOGICAL_OR] = "CPHVB_LOGICAL_OR",
    [CPHVB_LOGICAL_XOR] = "CPHVB_LOGICAL_XOR",
    [CPHVB_INVERT] = "CPHVB_INVERT",
    [CPHVB_LEFT_SHIFT] = "CPHVB_LEFT_SHIFT",
    [CPHVB_RIGHT_SHIFT] = "CPHVB_RIGHT_SHIFT",
    [CPHVB_GREATER] = "CPHVB_GREATER",
    [CPHVB_GREATER_EQUAL] = "CPHVB_GREATER_EQUAL",
    [CPHVB_LESS] = "CPHVB_LESS",
    [CPHVB_LESS_EQUAL] = "CPHVB_LESS_EQUAL",
    [CPHVB_NOT_EQUAL] = "CPHVB_NOT_EQUAL",
    [CPHVB_EQUAL] = "CPHVB_EQUAL",
    [CPHVB_MAXIMUM] = "CPHVB_MAXIMUM",
    [CPHVB_MINIMUM] = "CPHVB_MINIMUM",
    [CPHVB_ISFINITE] = "CPHVB_ISFINITE",
    [CPHVB_ISINF] = "CPHVB_ISINF",
    [CPHVB_ISNAN] = "CPHVB_ISNAN",
    [CPHVB_SIGNBIT] = "CPHVB_SIGNBIT",
    [CPHVB_MODF] = "CPHVB_MODF",
    [CPHVB_LDEXP] = "CPHVB_LDEXP",
    [CPHVB_FREXP] = "CPHVB_FREXP",
    [CPHVB_FLOOR] = "CPHVB_FLOOR",
    [CPHVB_CEIL] = "CPHVB_CEIL",
    [CPHVB_TRUNC] = "CPHVB_TRUNC",
    [CPHVB_LOG2] = "CPHVB_LOG2",
    [CPHVB_ISREAL] = "CPHVB_ISREAL",
    [CPHVB_ISCOMPLEX] = "CPHVB_ISCOMPLEX",
    [CPHVB_RELEASE] = "CPHVB_RELEASE",
    [CPHVB_SYNC] = "CPHVB_SYNC",
    [CPHVB_DISCARD] = "CPHVB_DISCARD",
    [CPHVB_DESTROY] = "CPHVB_DESTROY",
    [CPHVB_RANDOM] = "CPHVB_RANDOM",
    [CPHVB_ARANGE] = "CPHVB_ARANGE",
    [CPHVB_NONE] = "CPHVB_NONE"
};


/* Text string for operation
 *
 * @opcode Opcode for operation
 * @return Text string.
 */
static int first = 1;
char* _reduce_opcode_text;
const char* cphvb_opcode_text(cphvb_opcode opcode)
{
    if (first)
    {
        size_t str_size = 128;
        _reduce_opcode_text = (char*)malloc(str_size*CPHVB_NONE);
        for (int i = 0; i < CPHVB_NONE; ++i)
        {
            snprintf(_reduce_opcode_text+i*str_size,str_size,"%s_REDUCE",
                     _opcode_text[~CPHVB_REDUCE & i]);
        }
        first = 0;
    }
    if (CPHVB_REDUCE & opcode)
    {
        return _reduce_opcode_text+(~CPHVB_REDUCE & opcode);
    }
    else
    {
        return _opcode_text[opcode];
    }
}
