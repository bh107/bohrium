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

#include <bh_constant.h>
#include <bh_type.h>
#include <stdexcept>
#include <limits>


using namespace std;


int64_t bh_constant::get_int64() const
{
    switch(type) {
        case BH_INT8:
            return int64_t{value.int8};
        case BH_INT16:
            return int64_t{value.int16};
        case BH_INT32:
            return int64_t{value.int32};
        case BH_INT64:
            return value.int64;
        case BH_UINT8:
            return int64_t{value.uint8};
        case BH_UINT16:
            return int64_t{value.uint16};
        case BH_UINT32:
            return int64_t{value.uint32};
        case BH_UINT64:
            // We have to check that 'value' isn't too big for an int64_t
            if (value.uint64 < numeric_limits<int64_t>::max())
                return static_cast<int64_t>(value.uint64);
        default:
            throw overflow_error("Constant cannot be converted to int64_t");
    }
}

bool bh_constant::operator==(const bh_constant& other) const
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
