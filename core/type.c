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
 * along with cphVB.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "type.h"
#include "cphvb.h"

/* Data size in bytes for the different types */
const int _type_size[] = 
{ 
    [CPHVB_INT8] = 1,
    [CPHVB_INT16] = 2,
    [CPHVB_INT32] = 4,
    [CPHVB_INT64] = 8,
    [CPHVB_UINT8] = 1,
    [CPHVB_UINT16] = 2,
    [CPHVB_UINT32] = 4,
    [CPHVB_UINT64] = 8,
    [CPHVB_FLOAT16] = 2,
    [CPHVB_FLOAT32] = 4,
    [CPHVB_FLOAT64] = 8,
    [CPHVB_PTR] = sizeof(void*)
};


/* Byte size for type
 *
 * @type   Type code
 * @return Byte size
 */
int cphvb_type_size(cphvb_type type)
{
    return _type_size[type];
} 


const char* const _type_text[] = 
{ 
    [CPHVB_INT8] = "CPHVB_INT8",
    [CPHVB_INT16] = "CPHVB_INT16",
    [CPHVB_INT32] = "CPHVB_INT32",
    [CPHVB_INT64] = "CPHVB_INT64",
    [CPHVB_UINT8] = "CPHVB_UINT8",
    [CPHVB_UINT16] = "CPHVB_UINT16",
    [CPHVB_UINT32] = "CPHVB_UNIT32",
    [CPHVB_UINT64] = "CPHVB_UINT64",
    [CPHVB_FLOAT16] = "CPHVB_FLOAT16",
    [CPHVB_FLOAT32] = "CPHVB_FLOAT32",
    [CPHVB_FLOAT64] = "CPHVB_FLOAT64",
    [CPHVB_PTR] = "CPHVB_PTR"
};


/* Text string for type
 *
 * @type   Type code.
 * @return Text string.
 */
const char* cphvb_type_text(cphvb_type type)
{
    return _type_text[type];
} 
