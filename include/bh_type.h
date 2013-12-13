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

#ifndef __BH_TYPE_H
#define __BH_TYPE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Mapping of bohrium data types to C data types */
typedef unsigned char bh_bool;
typedef int8_t        bh_int8;
typedef int16_t       bh_int16;
typedef int32_t       bh_int32;
typedef int64_t       bh_int64;
typedef uint8_t       bh_uint8;
typedef uint16_t      bh_uint16;
typedef uint32_t      bh_uint32;
typedef uint64_t      bh_uint64;
typedef float         bh_float32;
typedef double        bh_float64;
typedef struct { float real, imag; } bh_complex64;
typedef struct { double real, imag; } bh_complex128;
typedef struct { bh_uint64 start, key; } bh_r123;

/* Codes for data types */
enum /* bh_type */
{
    BH_BOOL,
    BH_INT8,
    BH_INT16,
    BH_INT32,
    BH_INT64,
    BH_UINT8,
    BH_UINT16,
    BH_UINT32,
    BH_UINT64,
    BH_FLOAT32,
    BH_FLOAT64,
    BH_COMPLEX64,
    BH_COMPLEX128,
    BH_R123,
    BH_UNKNOWN
};

typedef int64_t    bh_intp;
typedef bh_intp    bh_index;
typedef bh_intp    bh_type;
typedef bh_intp    bh_opcode;
typedef bh_intp    bh_error;
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
    bh_float32    float32;
    bh_float64    float64;
    bh_complex64  complex64;
    bh_complex128 complex128;
    bh_r123       r123;
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
