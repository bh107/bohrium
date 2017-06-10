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

#include <stdint.h>

#include <bh_type.hpp>
#include <climits>
#include <cfloat>
#include <cassert>
#include <sstream>
#include <limits>

int bh_type_size(bh_type type)
{
    switch(type)
    {
        case bh_type::BOOL:       return  1;
        case bh_type::INT8:       return  1;
        case bh_type::INT16:      return  2;
        case bh_type::INT32:      return  4;
        case bh_type::INT64:      return  8;
        case bh_type::UINT8:      return  1;
        case bh_type::UINT16:     return  2;
        case bh_type::UINT32:     return  4;
        case bh_type::UINT64:     return  8;
        case bh_type::FLOAT32:    return  4;
        case bh_type::FLOAT64:    return  8;
        case bh_type::COMPLEX64:  return  8;
        case bh_type::COMPLEX128: return 16;
        case bh_type::R123:       return 16;
	}
    return -1;
}

const char* bh_type_text(bh_type type)
{
    switch(type)
    {
        case bh_type::BOOL:       return "BH_BOOL";
        case bh_type::INT8:       return "BH_INT8";
        case bh_type::INT16:      return "BH_INT16";
        case bh_type::INT32:      return "BH_INT32";
        case bh_type::INT64:      return "BH_INT64";
        case bh_type::UINT8:      return "BH_UINT8";
        case bh_type::UINT16:     return "BH_UINT16";
        case bh_type::UINT32:     return "BH_UINT32";
        case bh_type::UINT64:     return "BH_UINT64";
        case bh_type::FLOAT32:    return "BH_FLOAT32";
        case bh_type::FLOAT64:    return "BH_FLOAT64";
        case bh_type::COMPLEX64:  return "BH_COMPLEX64";
        case bh_type::COMPLEX128: return "BH_COMPLEX128";
        case bh_type::R123:       return "BH_R123";
    }
    return "UNKNOWN";
}

int bh_type_is_integer(bh_type type)
{
    switch(type)
    {
        case bh_type::UINT8:
        case bh_type::UINT16:
        case bh_type::UINT32:
        case bh_type::UINT64:
        case bh_type::INT8:
        case bh_type::INT16:
        case bh_type::INT32:
        case bh_type::INT64:
            return true;
        default:
            return false;
    }
}

int bh_type_is_unsigned_integer(bh_type type)
{
    switch(type)
    {
        case bh_type::UINT8:
        case bh_type::UINT16:
        case bh_type::UINT32:
        case bh_type::UINT64:
            return true;
        default:
            return false;
    }
}

int bh_type_is_signed_integer(bh_type type)
{
    switch(type)
    {
        case bh_type::INT8:
        case bh_type::INT16:
        case bh_type::INT32:
        case bh_type::INT64:
            return true;
        default:
            return false;
    }
}

int bh_type_is_float(bh_type type)
{
    switch (type) {
        case bh_type::FLOAT32:
        case bh_type::FLOAT64:
        case bh_type::COMPLEX64:
        case bh_type::COMPLEX128:
            return true;
        default:
            return false;
    }
}

int bh_type_is_complex(bh_type type)
{
    switch(type)
    {
        case bh_type::COMPLEX64:
        case bh_type::COMPLEX128:
            return true;
        default:
            return false;
    }
}

uint64_t bh_type_limit_max_integer(bh_type type)
{
    switch(type)
    {
        case bh_type::BOOL:   return 1;
        case bh_type::INT8:   return INT8_MAX;
        case bh_type::INT16:  return INT16_MAX;
        case bh_type::INT32:  return INT32_MAX;
        case bh_type::INT64:  return INT64_MAX;
        case bh_type::UINT8:  return UINT8_MAX;
        case bh_type::UINT16: return UINT16_MAX;
        case bh_type::UINT32: return UINT32_MAX;
        case bh_type::UINT64: return UINT64_MAX;
        default:
            assert(1 == 2);
            return 0;
    }
}

int64_t bh_type_limit_min_integer(bh_type type)
{
    switch(type)
    {
        case bh_type::BOOL:   return 1;
        case bh_type::INT8:   return INT8_MIN;
        case bh_type::INT16:  return INT16_MIN;
        case bh_type::INT32:  return INT32_MIN;
        case bh_type::INT64:  return INT64_MIN;
        case bh_type::UINT8:  return 0;
        case bh_type::UINT16: return 0;
        case bh_type::UINT32: return 0;
        case bh_type::UINT64: return 0;
        default:
            assert(1 == 2);
            return 0;
    }
}

double bh_type_limit_max_float(bh_type type)
{
    switch(type)
    {
        case bh_type::FLOAT32: return FLT_MAX_EXP;
        case bh_type::FLOAT64: return DBL_MAX_EXP;
        default:
            assert(1 == 2);
            return 0;
    }
}

double bh_type_limit_min_float(bh_type type)
{
    switch(type)
    {
        case bh_type::FLOAT32: return FLT_MIN_EXP;
        case bh_type::FLOAT64: return DBL_MIN_EXP;
        default:
            assert(1 == 2);
            return 0;
    }
}
