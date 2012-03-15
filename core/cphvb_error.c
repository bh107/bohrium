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

#ifdef __APPLE__
 //error.h is not required but the functions are known
 //the file is not found on OSX
#else
#include <error.h>
#endif
#include <cphvb.h>
#include <stdbool.h>

const char* _error_text[CPHVB_INST_NOT_SUPPORTED_FOR_SLICE+1];
bool _error_text_initialized = false;


/* Text string for error code
 *
 * @error  Error code.
 * @return Text string.
 */
const char* cphvb_error_text(cphvb_error error)
{
	if (!_error_text_initialized) {
		_error_text[CPHVB_SUCCESS] = "CPHVB_SUCCESS";
		_error_text[CPHVB_ERROR] = "CPHVB_ERROR";
		_error_text[CPHVB_TYPE_ERROR] = "CPHVB_TYPE_ERROR";
		_error_text[CPHVB_TYPE_NOT_SUPPORTED] = "CPHVB_TYPE_NOT_SUPPORTED";
		_error_text[CPHVB_TYPE_NOT_SUPPORTED_BY_OP] ="CPHVB_TYPE_NOT_SUPPORTED_BY_OP";
		_error_text[CPHVB_TYPE_COMB_NOT_SUPPORTED] = "CPHVB_TYPE_COMB_NOT_SUPPORTED";
		_error_text[CPHVB_OUT_OF_MEMORY] = "CPHVB_OUT_OF_MEMORY";
		_error_text[CPHVB_RESULT_IS_CONSTANT] = "CPHVB_RESULT_IS_CONSTANT";
		_error_text[CPHVB_OPERAND_UNKNOWN] = "CPHVB_OPERAND_UNKNOWN";
		_error_text[CPHVB_ALREADY_INITALIZED] = "CPHVB_ALREADY_INITALIZED";
		_error_text[CPHVB_NOT_INITALIZED] = "CPHVB_NOT_INITALIZED",
		_error_text[CPHVB_PARTIAL_SUCCESS] = "CPHVB_PARTIAL_SUCCESS";
		_error_text[CPHVB_INST_NOT_SUPPORTED] = "CPHVB_INST_NOT_SUPPORTED";
		_error_text[CPHVB_INST_NOT_SUPPORTED_FOR_SLICE] = "CPHVB_INST_NOT_SUPPORTED_FOR_SLICE";
		_error_text[CPHVB_INST_DONE] = "CPHVB_INST_DONE";
		_error_text[CPHVB_INST_UNDONE] = "CPHVB_INST_UNDONE";
		_error_text_initialized = true;
	}
    return _error_text[error];
}
