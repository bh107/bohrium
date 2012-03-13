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

#include <cphvb_type.h>
#include <cphvb.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Data size in bytes for the different types */
int _type_size[CPHVB_UNKNOWN+1];
bool _type_size_initialized = false;

/* Text string for the different types */
const char* _type_text[CPHVB_UNKNOWN+1];
bool _type_text_initialized = false;

/* Byte size for type
 *
 * @type   Type code
 * @return Byte size
 */
int cphvb_type_size(cphvb_type type)
{
	if (!_type_size_initialized) {
		_type_size[CPHVB_BOOL] = 1;
		_type_size[CPHVB_INT8] = 1;
		_type_size[CPHVB_INT16] = 2;
		_type_size[CPHVB_INT32] = 4;
		_type_size[CPHVB_INT64] = 8;
		_type_size[CPHVB_UINT8] = 1;
		_type_size[CPHVB_UINT16] = 2;
		_type_size[CPHVB_UINT32] = 4;
		_type_size[CPHVB_UINT64] = 8;
		_type_size[CPHVB_FLOAT16] = 2;
		_type_size[CPHVB_FLOAT32] = 4;
		_type_size[CPHVB_FLOAT64] = 8;
		_type_size[CPHVB_UNKNOWN] = -1;
		_type_size_initialized = true;
	}
    return _type_size[type];
}

/* Text string for type
 *
 * @type   Type code.
 * @return Text string.
 */
const char* cphvb_type_text(cphvb_type type)
{
	if (!_type_text_initialized) {
		_type_text[CPHVB_BOOL]    = "CPHVB_BOOL";
		_type_text[CPHVB_INT8]    = "CPHVB_INT8";
		_type_text[CPHVB_INT16]   = "CPHVB_INT16";
		_type_text[CPHVB_INT32]   = "CPHVB_INT32";
		_type_text[CPHVB_INT64]   = "CPHVB_INT64";
		_type_text[CPHVB_UINT8]   = "CPHVB_UINT8";
		_type_text[CPHVB_UINT16]  = "CPHVB_UINT16";
		_type_text[CPHVB_UINT32]  = "CPHVB_UNIT32";
		_type_text[CPHVB_UINT64]  = "CPHVB_UINT64";
		_type_text[CPHVB_FLOAT16] = "CPHVB_FLOAT16";
		_type_text[CPHVB_FLOAT32] = "CPHVB_FLOAT32";
		_type_text[CPHVB_FLOAT64] = "CPHVB_FLOAT64";
		_type_text[CPHVB_UNKNOWN] = "CPHVB_UNKNOWN";
		_type_text_initialized = true;
	}
    return _type_text[type];
}

#ifdef __cplusplus
}
#endif
