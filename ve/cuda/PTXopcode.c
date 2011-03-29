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

#include <limits.h>
#include "PTXopcode.h"

const int _ptxOperands[] =
{
    [PTX_BRA] = 1, 
    [PTX_ADD] = 3,
    [PTX_SUB] = 3,
    [PTX_MUL] = 3,
    [PTX_MAD] = 4,
    [PTX_MAD_WIDE] = 4,
    [PTX_DIV] = 3,
    [PTX_REM] = 3,
    [PTX_ST_GLOBAL] = 3,
    [PTX_LD_GLOBAL] = 3,
    [PTX_LD_PARAM] = 2,
    [PTX_MOV] = 2,
    [PTX_SET_EQ] = 3,
    [PTX_SET_NE] = 3,
    [PTX_SET_LT] = 3,
    [PTX_SET_LE] = 3,
    [PTX_SET_GT] = 3,
    [PTX_SET_GE] = 3,
    [PTX_SETP_GE] = 3,
    [PTX_CVT] = 2,
    [PTX_MEMBAR] = 0,
    [PTX_EXIT] = 0
};

const int ptxOperands(PTXopcode opcode)
{
    return _ptxOperands[opcode];
}

const int ptxSrcOperands(PTXopcode opcode)
{
    return _ptxOperands[opcode]-1;
}

const int _ptxOpcodeMap[] =
{
    [CPHVB_ADD] = PTX_ADD,
    [CPHVB_SUBTRACT] = PTX_SUB,
    [CPHVB_MULTIPLY] = PTX_MUL,
    [CPHVB_DIVIDE] = PTX_DIV,
    [CPHVB_LOGADDEXP] = -1,
    [CPHVB_LOGADDEXP2] = -1,
    [CPHVB_TRUE_DIVIDE] = -1,
    [CPHVB_FLOOR_DIVIDE] = -1,
    [CPHVB_NEGATIVE] = -1,
    [CPHVB_POWER] = -1,
    [CPHVB_REMAINDER] = -1,
    [CPHVB_MOD] = PTX_REM,
    [CPHVB_FMOD] = -1,
    [CPHVB_ABSOLUTE] = -1,
    [CPHVB_RINT] = -1,
    [CPHVB_SIGN] = -1,
    [CPHVB_CONJ] = -1,
    [CPHVB_EXP] = -1,
    [CPHVB_EXP2] = -1,
    [CPHVB_LOG] = -1,
    [CPHVB_LOG10] = -1,
    [CPHVB_EXPM1] = -1,
    [CPHVB_LOG1P] = -1,
    [CPHVB_SQRT] = -1,
    [CPHVB_SQUARE] = -1,
    [CPHVB_RECIPROCAL] = -1,
    [CPHVB_ONES_LIKE] = -1,
    [CPHVB_SIN] = -1,
    [CPHVB_COS] = -1,
    [CPHVB_TAN] = -1,
    [CPHVB_ARCSIN] = -1,
    [CPHVB_ARCCOS] = -1,
    [CPHVB_ARCTAN] = -1,
    [CPHVB_ARCTAN2] = -1,
    [CPHVB_HYPOT] = -1,
    [CPHVB_SINH] = -1,
    [CPHVB_COSH] = -1,
    [CPHVB_TANH] = -1,
    [CPHVB_ARCSINH] = -1,
    [CPHVB_ARCCOSH] = -1,
    [CPHVB_ARCTANH] = -1,
    [CPHVB_DEG2RAD] = -1,
    [CPHVB_RAD2DEG] = -1,
    [CPHVB_BITWISE_AND] = -1,
    [CPHVB_BITWISE_OR] = -1,
    [CPHVB_BITWISE_XOR] = -1,
    [CPHVB_LOGICAL_NOT] = -1,
    [CPHVB_LOGICAL_AND] = -1,
    [CPHVB_LOGICAL_OR] = -1,
    [CPHVB_LOGICAL_XOR] = -1,
    [CPHVB_INVERT] = -1,
    [CPHVB_LEFT_SHIFT] = -1,
    [CPHVB_RIGHT_SHIFT] = -1,
    [CPHVB_GREATER] = PTX_SET_GT,
    [CPHVB_GREATER_EQUAL] = PTX_SET_GE,
    [CPHVB_LESS] = PTX_SET_LT,
    [CPHVB_LESS_EQUAL] = PTX_SET_LE,
    [CPHVB_NOT_EQUAL] = PTX_SET_NE,
    [CPHVB_EQUAL] = PTX_SET_EQ,
    [CPHVB_MAXIMUM] = -1,
    [CPHVB_MINIMUM] = -1,
    [CPHVB_ISFINITE] = -1,
    [CPHVB_ISINF] = -1,
    [CPHVB_ISNAN] = -1,
    [CPHVB_SIGNBIT] = -1,
    [CPHVB_MODF] = -1,
    [CPHVB_LDEXP] = -1,
    [CPHVB_FREXP] = -1,
    [CPHVB_FLOOR] = -1,
    [CPHVB_CEIL] = -1,
    [CPHVB_TRUNC] = -1,
    [CPHVB_LOG2] = -1,
    [CPHVB_ISREAL] = -1,
    [CPHVB_ISCOMPLEX] = -1,
    [CPHVB_RELEASE] = PTX_NONE,
    [CPHVB_SYNC] = PTX_NONE,
    [CPHVB_DISCARD] = PTX_NONE,
    [CPHVB_DESTORY] = -1,
    [CPHVB_RANDOM] = PTX_NONE,
    [CPHVB_ARANGE] = -1,
    [CPHVB_NONE] = -1
};

PTXopcode ptxOpcodeMap(cphvb_opcode opcode)
{
    return _ptxOpcodeMap[opcode];
}
