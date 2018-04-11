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
#pragma once

#include <stdexcept>
#include <complex>
#include <stdint.h>

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
enum class bh_type
{
    BOOL,
    INT8,
    INT16,
    INT32,
    INT64,
    UINT8,
    UINT16,
    UINT32,
    UINT64,
    FLOAT32,
    FLOAT64,
    COMPLEX64,
    COMPLEX128,
    R123
};

// Return a `bh_type` based on a template type
template<typename T> inline bh_type bh_type_from_template() {
    throw std::runtime_error("Type not supported in Bohrium");
}
template<> inline bh_type bh_type_from_template<bool>() {
    return bh_type::BOOL;
}
template<> inline bh_type bh_type_from_template<int8_t>() {
    return bh_type::INT8;
}
template<> inline bh_type bh_type_from_template<int16_t>() {
    return bh_type::INT16;
}
template<> inline bh_type bh_type_from_template<int32_t>() {
    return bh_type::INT32;
}
template<> inline bh_type bh_type_from_template<int64_t>() {
    return bh_type::INT64;
}
template<> inline bh_type bh_type_from_template<uint8_t>() {
    return bh_type::UINT8;
}
template<> inline bh_type bh_type_from_template<uint16_t>() {
    return bh_type::UINT16;
}
template<> inline bh_type bh_type_from_template<uint32_t>() {
    return bh_type::UINT32;
}
template<> inline bh_type bh_type_from_template<uint64_t>() {
    return bh_type::UINT64;
}
template<> inline bh_type bh_type_from_template<float>() {
    return bh_type::FLOAT32;
}
template<> inline bh_type bh_type_from_template<double>() {
    return bh_type::FLOAT64;
}
template<> inline bh_type bh_type_from_template<std::complex<float> >() {
    return bh_type::COMPLEX64;
}
template<> inline bh_type bh_type_from_template<std::complex<double> >() {
    return bh_type::COMPLEX128;
}
template<> inline bh_type bh_type_from_template<bh_r123>() {
    return bh_type::R123;
}

typedef int64_t    bh_opcode;


/* Byte size for type
 *
 * @type   Type code
 * @return Byte size
 */
int bh_type_size(bh_type type);

/* Text string for type
 *
 * @type   Type code.
 * @return Text string.
 */
const char* bh_type_text(bh_type type);

/* Is type an integer type
 *
 * @type   The type.
 * @return 1 if integer type else 0.
 */
int bh_type_is_integer(bh_type type);

/* Is type an signed integer type
 *
 * @type   The type.
 * @return 1 if true else 0.
 */
int bh_type_is_signed_integer(bh_type type);

/* Is type an unsigned integer type
 *
 * @type   The type.
 * @return 1 if true else 0.
 */
int bh_type_is_unsigned_integer(bh_type type);

/* Is type an float type
 *
 * @type   The type.
 * @return 1 if integer type else 0.
 */
int bh_type_is_float(bh_type type);

/* Is type an complex type
 *
 * @type   The type.
 * @return 1 if integer type else 0.
 */
int bh_type_is_complex(bh_type type);

/* Maximum value of integer type (incl. boolean)
 *
 * @type   The type.
 */
uint64_t bh_type_limit_max_integer(bh_type type);

/* Minimum value of integer type (incl. boolean)
 *
 * @type   The type.
 */
int64_t bh_type_limit_min_integer(bh_type type);

/* Maximum value of float type (excl. complex)
 *
 * @type   The type.
 */
double bh_type_limit_max_float(bh_type type);

/* Minimum value of float type (excl. complex)
 *
 * @type   The type.
 */
double bh_type_limit_min_float(bh_type type);
