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

#ifndef __BH_CONSTANT_H
#define __BH_CONSTANT_H

union bh_constant_value
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
};

struct bh_constant
{
    bh_constant_value value;
    bh_type type;

    bool operator==(const bh_constant& other) const
    {
        if (other.type != type) return false;

        switch (type) {
            case BH_BOOL:
                return other.value.bool8 == value.bool8;
            case BH_INT8:
                return other.value.int8 == value.int8;
            case BH_INT16:
                return other.value.int16 == value.int16;
            case BH_INT32:
                return other.value.int32 == value.int32;
            case BH_INT64:
                return other.value.int64 == value.int64;
            case BH_UINT8:
                return other.value.uint8 == value.uint8;
            case BH_UINT16:
                return other.value.uint16 == value.uint16;
            case BH_UINT32:
                return other.value.uint32 == value.uint32;
            case BH_UINT64:
                return other.value.uint64 == value.uint64;
            case BH_FLOAT32:
                return other.value.float32 == value.float32;
            case BH_FLOAT64:
                return other.value.float64 == value.float64;
            case BH_COMPLEX64:
                return other.value.complex64.real == value.complex64.real &&
                       other.value.complex64.imag == value.complex64.imag;
            case BH_COMPLEX128:
                return other.value.complex128.real == value.complex128.real &&
                       other.value.complex128.imag == value.complex128.imag;
            case BH_R123:
                return other.value.r123.start == value.r123.start &&
                       other.value.r123.key == value.r123.key;
            case BH_UNKNOWN:
            default:
                return false;
        }
    }

    bool operator!=(const bh_constant& other) const
    {
        return !(other == *this);
    }
};

#endif
