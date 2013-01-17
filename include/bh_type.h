/*
This file is part of Bohrium and copyright (c) 2012 the Bohrium
team <http://www.bh107.org>.

Bohrium is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as 
published by the Free Software Foundation, either version 3 
of the License, or (at your option) any later version.

Bohrium is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the 
GNU Lesser General Public License along with Bohrium. 

If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __CPHVB_TYPE_H
#define __CPHVB_TYPE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Mapping of cphvb data types to C data types */
typedef unsigned char bh_bool;
typedef int8_t        bh_int8;
typedef int16_t       bh_int16;
typedef int32_t       bh_int32;
typedef int64_t       bh_int64;
typedef uint8_t       bh_uint8;
typedef uint16_t      bh_uint16;
typedef uint32_t      bh_uint32;
typedef uint64_t      bh_uint64;
typedef uint16_t      bh_float16;
typedef float         bh_float32;
typedef double        bh_float64;
typedef struct { float real, imag; } bh_complex64;
typedef struct { double real, imag; } bh_complex128;


/* Codes for data types */
enum /* bh_type */
{
    CPHVB_BOOL,
    CPHVB_INT8,
    CPHVB_INT16,
    CPHVB_INT32,
    CPHVB_INT64,
    CPHVB_UINT8,
    CPHVB_UINT16,
    CPHVB_UINT32,
    CPHVB_UINT64,
    CPHVB_FLOAT16,
    CPHVB_FLOAT32,
    CPHVB_FLOAT64,
    CPHVB_COMPLEX64,
    CPHVB_COMPLEX128,
    CPHVB_UNKNOWN
};

typedef int64_t    bh_intp;
typedef bh_intp bh_index;
typedef bh_intp bh_type;
typedef bh_intp bh_opcode;
typedef bh_intp bh_error;
typedef void*      bh_data_ptr;

typedef union /* bh_constant_value */
{
    bh_bool       bool8;
    bh_int8       int8;
    bh_int16      int16;
    bh_int32      int32;
    bh_int64      int64;
    bh_uint8      uint8;
    bh_uint16     uint16;
    bh_uint32     uint32;
    bh_uint64     uint64;
    bh_float16    float16;
    bh_float32    float32;
    bh_float64    float64;
    bh_complex64  complex64;
    bh_complex128 complex128;
} bh_constant_value;

typedef struct
{
    bh_constant_value value;
    bh_type type;
} bh_constant;

#ifdef __cplusplus
}
#endif

#endif
