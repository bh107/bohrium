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
#include <bh_win.h>

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
typedef void*      bh_data_ptr;


/* Byte size for type
 *
 * @type   Type code
 * @return Byte size
 */
DLLEXPORT int bh_type_size(bh_type type);

/* Text string for type
 *
 * @type   Type code.
 * @return Text string.
 */
DLLEXPORT const char* bh_type_text(bh_type type);

/* Is type an integer type
 *
 * @type   The type.
 * @return 1 if integer type else 0.
 */
DLLEXPORT int bh_type_is_integer(bh_type type);

/* Is type an signed integer type
 *
 * @type   The type.
 * @return 1 if true else 0.
 */
DLLEXPORT int bh_type_is_signed_integer(bh_type type);

/* Maximum value of integer type (incl. boolean)
 *
 * @type   The type.
 */
DLLEXPORT uint64_t bh_type_limit_max_integer(bh_type type);

/* Minimum value of integer type (incl. boolean)
 *
 * @type   The type.
 */
DLLEXPORT int64_t bh_type_limit_min_integer(bh_type type);

/* Maximum value of float type (excl. complex)
 *
 * @type   The type.
 */
DLLEXPORT double bh_type_limit_max_float(bh_type type);

/* Minimum value of float type (excl. complex)
 *
 * @type   The type.
 */
DLLEXPORT double bh_type_limit_min_float(bh_type type);

#endif
