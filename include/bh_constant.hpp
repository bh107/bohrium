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

#include <iostream>

#include <bh_type.h>

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

    //Convert the constant value to an int64
    //Throw an overflow_error() exception if impossible
    int64_t get_int64() const;

    //Convert the constant value to an double
    //Throw an overflow_error() exception if impossible
    //Throw an runtime_error() exception if type is unknown
    double get_double() const;

    //Convert the constant value to an double
    //Throw an overflow_error() exception if impossible
    //Throw an runtime_error() exception if type is unknown
    void set_double(double value);

    bool operator==(const bh_constant& other) const;

    bool operator!=(const bh_constant& other) const
    {
        return !(other == *this);
    }
};

//Implements pprint of a constant
DLLEXPORT std::ostream& operator<<(std::ostream& out, const bh_constant& constant);

#endif
