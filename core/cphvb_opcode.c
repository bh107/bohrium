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


/* Number of operands for a given operation */
int _operands[CPHVB_NONE+1];
bool _operands_intialized = false;

/* Number of operands for operation
 *
 * @opcode Opcode for operation
 * @return Number of operands
 */
int cphvb_operands(cphvb_opcode opcode)
{
    // The reduce family of operations take the same number of arguments
    // as the regular operation
    
    if (!_operands_intialized) {
		_operands[CPHVB_ADD] = 3;
		_operands[CPHVB_SUBTRACT] = 3;
		_operands[CPHVB_MULTIPLY] = 3;
		_operands[CPHVB_DIVIDE] = 3;
		_operands[CPHVB_LOGADDEXP] = 3;
		_operands[CPHVB_LOGADDEXP2] = 3;
		_operands[CPHVB_TRUE_DIVIDE] = 3;
		_operands[CPHVB_FLOOR_DIVIDE] = 3;
		_operands[CPHVB_NEGATIVE] = 2;
		_operands[CPHVB_POWER] = 3;
		_operands[CPHVB_REMAINDER] = 3;
		_operands[CPHVB_MOD] = 3;
		_operands[CPHVB_FMOD] = 3;
		_operands[CPHVB_ABSOLUTE] = 2;
		_operands[CPHVB_RINT] = 2;
		_operands[CPHVB_SIGN] = 2;
		_operands[CPHVB_CONJ] = 2;
		_operands[CPHVB_EXP] = 2;
		_operands[CPHVB_EXP2] = 2;
		_operands[CPHVB_LOG] = 2;
		_operands[CPHVB_LOG10] = 2;
		_operands[CPHVB_EXPM1] = 2;
		_operands[CPHVB_LOG1P] = 2;
		_operands[CPHVB_SQRT] = 2;
		_operands[CPHVB_SQUARE] = 2;
		_operands[CPHVB_RECIPROCAL] = 2;
		_operands[CPHVB_ONES_LIKE] = 2;
		_operands[CPHVB_SIN] = 2;
		_operands[CPHVB_COS] = 2;
		_operands[CPHVB_TAN] = 2;
		_operands[CPHVB_ARCSIN] = 2;
		_operands[CPHVB_ARCCOS] = 2;
		_operands[CPHVB_ARCTAN] = 2;
		_operands[CPHVB_ARCTAN2] = 3;
		_operands[CPHVB_HYPOT] = 3;
		_operands[CPHVB_SINH] = 2;
		_operands[CPHVB_COSH] = 2;
		_operands[CPHVB_TANH] = 2;
		_operands[CPHVB_ARCSINH] = 2;
		_operands[CPHVB_ARCCOSH] = 2;
		_operands[CPHVB_ARCTANH] = 2;
		_operands[CPHVB_DEG2RAD] = 2;
		_operands[CPHVB_RAD2DEG] = 2;
		_operands[CPHVB_BITWISE_AND] = 3;
		_operands[CPHVB_BITWISE_OR] = 3;
		_operands[CPHVB_BITWISE_XOR] = 3;
		_operands[CPHVB_LOGICAL_NOT] = 2;
		_operands[CPHVB_LOGICAL_AND] = 3;
		_operands[CPHVB_LOGICAL_OR] = 3;
		_operands[CPHVB_LOGICAL_XOR] = 3;
		_operands[CPHVB_INVERT] = 2;
		_operands[CPHVB_LEFT_SHIFT] = 3;
		_operands[CPHVB_RIGHT_SHIFT] = 3;
		_operands[CPHVB_GREATER] = 3;
		_operands[CPHVB_GREATER_EQUAL] = 3;
		_operands[CPHVB_LESS] = 3;
		_operands[CPHVB_LESS_EQUAL] = 3;
		_operands[CPHVB_NOT_EQUAL] = 3;
		_operands[CPHVB_EQUAL] = 3;
		_operands[CPHVB_MAXIMUM] = 3;
		_operands[CPHVB_MINIMUM] = 3;
		_operands[CPHVB_ISFINITE] = 2;
		_operands[CPHVB_ISINF] = 2;
		_operands[CPHVB_ISNAN] = 2;
		_operands[CPHVB_SIGNBIT] = 2;
		_operands[CPHVB_MODF] = 3;
		_operands[CPHVB_LDEXP] = 3;
		_operands[CPHVB_FREXP] = 3;
		_operands[CPHVB_FLOOR] = 2;
		_operands[CPHVB_CEIL] = 2;
		_operands[CPHVB_TRUNC] = 2;
		_operands[CPHVB_LOG2] = 2;
		_operands[CPHVB_ISREAL] = 2;
		_operands[CPHVB_ISCOMPLEX] = 2;
		_operands[CPHVB_RELEASE] = 1;
		_operands[CPHVB_SYNC] = 1;
		_operands[CPHVB_DISCARD] = 1;
		_operands[CPHVB_DESTROY] = 1;
		_operands[CPHVB_RANDOM] = 1;
		_operands[CPHVB_ARANGE] = 3;
		_operands[CPHVB_IDENTITY] = 2;
		_operands[CPHVB_USERFUNC] = 0;
		_operands[CPHVB_NONE] = 0; 	
    	_operands_intialized = true;
    }
    
    return _operands[opcode];
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
	if (!_opcode_text_initialized) {
		_opcode_text[CPHVB_ADD] = "CPHVB_ADD";
		_opcode_text[CPHVB_SUBTRACT] = "CPHVB_SUBTRACT";
		_opcode_text[CPHVB_MULTIPLY] = "CPHVB_MULTIPLY";
		_opcode_text[CPHVB_DIVIDE] = "CPHVB_DIVIDE";
		_opcode_text[CPHVB_LOGADDEXP] = "CPHVB_LOGADDEXP";
		_opcode_text[CPHVB_LOGADDEXP2] = "CPHVB_LOGADDEXP2";
		_opcode_text[CPHVB_TRUE_DIVIDE] = "CPHVB_TRUE_DIVIDE";
		_opcode_text[CPHVB_FLOOR_DIVIDE] = "CPHVB_FLOOR_DIVIDE";
		_opcode_text[CPHVB_NEGATIVE] = "CPHVB_NEGATIVE";
		_opcode_text[CPHVB_POWER] = "CPHVB_POWER";
		_opcode_text[CPHVB_REMAINDER] = "CPHVB_REMAINDER";
		_opcode_text[CPHVB_MOD] = "CPHVB_MOD";
		_opcode_text[CPHVB_FMOD] = "CPHVB_FMOD";
		_opcode_text[CPHVB_ABSOLUTE] = "CPHVB_ABSOLUTE";
		_opcode_text[CPHVB_RINT] = "CPHVB_RINT";
		_opcode_text[CPHVB_SIGN] = "CPHVB_SIGN";
		_opcode_text[CPHVB_CONJ] = "CPHVB_CONJ";
		_opcode_text[CPHVB_EXP] = "CPHVB_EXP";
		_opcode_text[CPHVB_EXP2] = "CPHVB_EXP2";
		_opcode_text[CPHVB_LOG] = "CPHVB_LOG";
		_opcode_text[CPHVB_LOG10] = "CPHVB_LOG10";
		_opcode_text[CPHVB_EXPM1] = "CPHVB_EXPM1";
		_opcode_text[CPHVB_LOG1P] = "CPHVB_LOG1P";
		_opcode_text[CPHVB_SQRT] = "CPHVB_SQRT";
		_opcode_text[CPHVB_SQUARE] = "CPHVB_SQUARE";
		_opcode_text[CPHVB_RECIPROCAL] = "CPHVB_RECIPROCAL";
		_opcode_text[CPHVB_ONES_LIKE] = "CPHVB_ONES_LIKE";
		_opcode_text[CPHVB_SIN] = "CPHVB_SIN";
		_opcode_text[CPHVB_COS] = "CPHVB_COS";
		_opcode_text[CPHVB_TAN] = "CPHVB_TAN";
		_opcode_text[CPHVB_ARCSIN] = "CPHVB_ARCSIN";
		_opcode_text[CPHVB_ARCCOS] = "CPHVB_ARCCOS";
		_opcode_text[CPHVB_ARCTAN] = "CPHVB_ARCTAN";
		_opcode_text[CPHVB_ARCTAN2] = "CPHVB_ARCTAN2";
		_opcode_text[CPHVB_HYPOT] = "CPHVB_HYPOT";
		_opcode_text[CPHVB_SINH] = "CPHVB_SINH";
		_opcode_text[CPHVB_COSH] = "CPHVB_COSH";
		_opcode_text[CPHVB_TANH] = "CPHVB_TANH";
		_opcode_text[CPHVB_ARCSINH] = "CPHVB_ARCSINH";
		_opcode_text[CPHVB_ARCCOSH] = "CPHVB_ARCCOSH";
		_opcode_text[CPHVB_ARCTANH] = "CPHVB_ARCTANH";
		_opcode_text[CPHVB_DEG2RAD] = "CPHVB_DEG2RAD";
		_opcode_text[CPHVB_RAD2DEG] = "CPHVB_RAD2DEG";
		_opcode_text[CPHVB_BITWISE_AND] = "CPHVB_BITWISE_AND";
		_opcode_text[CPHVB_BITWISE_OR] = "CPHVB_BITWISE_OR";
		_opcode_text[CPHVB_BITWISE_XOR] = "CPHVB_BITWISE_XOR";
		_opcode_text[CPHVB_LOGICAL_NOT] = "CPHVB_LOGICAL_NOT";
		_opcode_text[CPHVB_LOGICAL_AND] = "CPHVB_LOGICAL_AND";
		_opcode_text[CPHVB_LOGICAL_OR] = "CPHVB_LOGICAL_OR";
		_opcode_text[CPHVB_LOGICAL_XOR] = "CPHVB_LOGICAL_XOR";
		_opcode_text[CPHVB_INVERT] = "CPHVB_INVERT";
		_opcode_text[CPHVB_LEFT_SHIFT] = "CPHVB_LEFT_SHIFT";
		_opcode_text[CPHVB_RIGHT_SHIFT] = "CPHVB_RIGHT_SHIFT";
		_opcode_text[CPHVB_GREATER] = "CPHVB_GREATER";
		_opcode_text[CPHVB_GREATER_EQUAL] = "CPHVB_GREATER_EQUAL";
		_opcode_text[CPHVB_LESS] = "CPHVB_LESS";
		_opcode_text[CPHVB_LESS_EQUAL] = "CPHVB_LESS_EQUAL";
		_opcode_text[CPHVB_NOT_EQUAL] = "CPHVB_NOT_EQUAL";
		_opcode_text[CPHVB_EQUAL] = "CPHVB_EQUAL";
		_opcode_text[CPHVB_MAXIMUM] = "CPHVB_MAXIMUM";
		_opcode_text[CPHVB_MINIMUM] = "CPHVB_MINIMUM";
		_opcode_text[CPHVB_ISFINITE] = "CPHVB_ISFINITE";
		_opcode_text[CPHVB_ISINF] = "CPHVB_ISINF";
		_opcode_text[CPHVB_ISNAN] = "CPHVB_ISNAN";
		_opcode_text[CPHVB_SIGNBIT] = "CPHVB_SIGNBIT";
		_opcode_text[CPHVB_MODF] = "CPHVB_MODF";
		_opcode_text[CPHVB_LDEXP] = "CPHVB_LDEXP";
		_opcode_text[CPHVB_FREXP] = "CPHVB_FREXP";
		_opcode_text[CPHVB_FLOOR] = "CPHVB_FLOOR";
		_opcode_text[CPHVB_CEIL] = "CPHVB_CEIL";
		_opcode_text[CPHVB_TRUNC] = "CPHVB_TRUNC";
		_opcode_text[CPHVB_LOG2] = "CPHVB_LOG2";
		_opcode_text[CPHVB_ISREAL] = "CPHVB_ISREAL";
		_opcode_text[CPHVB_ISCOMPLEX] = "CPHVB_ISCOMPLEX";
		_opcode_text[CPHVB_RELEASE] = "CPHVB_RELEASE";
		_opcode_text[CPHVB_SYNC] = "CPHVB_SYNC";
		_opcode_text[CPHVB_DISCARD] = "CPHVB_DISCARD";
		_opcode_text[CPHVB_DESTROY] = "CPHVB_DESTROY";
		_opcode_text[CPHVB_RANDOM] = "CPHVB_RANDOM";
		_opcode_text[CPHVB_ARANGE] = "CPHVB_ARANGE";
		_opcode_text[CPHVB_IDENTITY] = "CPHVB_IDENTITY";
		_opcode_text[CPHVB_USERFUNC] = "CPHVB_USERFUNC";
		_opcode_text[CPHVB_NONE] = "CPHVB_NONE";
		_opcode_text_initialized = true;
	}
    return _opcode_text[opcode];
}
