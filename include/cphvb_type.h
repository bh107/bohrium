/*
This file is part of cphVB and copyright (c) 2012 the cphVB team:
http://cphvb.bitbucket.org

cphVB is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as 
published by the Free Software Foundation, either version 3 
of the License, or (at your option) any later version.

cphVB is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the 
GNU Lesser General Public License along with cphVB. 

If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __CPHVB_TYPE_H
#define __CPHVB_TYPE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Mapping of cphvb data types to C data types */
typedef unsigned char cphvb_bool;
typedef int8_t        cphvb_int8;
typedef int16_t       cphvb_int16;
typedef int32_t       cphvb_int32;
typedef int64_t       cphvb_int64;
typedef uint8_t       cphvb_uint8;
typedef uint16_t      cphvb_uint16;
typedef uint32_t      cphvb_uint32;
typedef uint64_t      cphvb_uint64;
typedef uint16_t      cphvb_float16;
typedef float         cphvb_float32;
typedef double        cphvb_float64;
typedef struct { float real, imag; } cphvb_complex64;
typedef struct { double real, imag; } cphvb_complex128;


/* Codes for data types */
enum /* cphvb_type */
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

typedef int64_t    cphvb_intp;
typedef cphvb_intp cphvb_index;
typedef cphvb_intp cphvb_type;
typedef cphvb_intp cphvb_opcode;
typedef cphvb_intp cphvb_error;
typedef void*      cphvb_data_ptr;

typedef union /* cphvb_constant_value */
{
    cphvb_bool       bool8;
    cphvb_int8       int8;
    cphvb_int16      int16;
    cphvb_int32      int32;
    cphvb_int64      int64;
    cphvb_uint8      uint8;
    cphvb_uint16     uint16;
    cphvb_uint32     uint32;
    cphvb_uint64     uint64;
    cphvb_float16    float16;
    cphvb_float32    float32;
    cphvb_float64    float64;
    cphvb_complex64  complex64;
    cphvb_complex128 complex128;
} cphvb_constant_value;

typedef struct
{
    cphvb_constant_value value;
    cphvb_type type;
} cphvb_constant;

#ifdef __cplusplus
}
#endif

#endif
