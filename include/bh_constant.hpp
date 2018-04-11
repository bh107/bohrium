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

#include <iostream>
#include <bh_type.hpp>

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

    // Constructors for each possible union type
    bh_constant_value() = default;
    bh_constant_value(bool val) : bool8(val) {}
    bh_constant_value(int8_t val) : int8(val) {}
    bh_constant_value(int16_t val) : int16(val) {}
    bh_constant_value(int32_t val) : int32(val) {}
    bh_constant_value(int64_t val) : int64(val) {}
    bh_constant_value(uint8_t val) : uint8(val) {}
    bh_constant_value(uint16_t val) : uint16(val) {}
    bh_constant_value(uint32_t val) : uint32(val) {}
    bh_constant_value(uint64_t val) : uint64(val) {}
    bh_constant_value(float val) : float32(val) {}
    bh_constant_value(double val) : float64(val) {}
    bh_constant_value(std::complex<float> val) : complex64{val.real(), val.imag()} {}
    bh_constant_value(std::complex<double> val) : complex128{val.real(), val.imag()} {}
    bh_constant_value(bh_r123 val) : r123(val) {}
};

class bh_constant
{
public:
    bh_constant_value value;
    bh_type type;

    bh_constant() = default;

    //Constructor based on type
    template<typename T>
    bh_constant(T val) : value(val), type(bh_type_from_template<T>()){}

    //Convert the constant value to an int64
    //Throw an overflow_error() exception if impossible
    int64_t get_int64() const;

    //Convert the constant value to an uint64
    //Throw an overflow_error() exception if impossible
    uint64_t get_uint64() const;

    //Convert the constant value to an double
    //Throw an overflow_error() exception if impossible
    //Throw an runtime_error() exception if type is unknown
    double get_double() const;

    //Set the constant based on value
    //Throw an overflow_error() exception if impossible
    //Throw an runtime_error() exception if type is unknown
    void set_double(double value);

    bool operator==(const bh_constant& other) const;

    bool operator!=(const bh_constant& other) const
    {
        return !(other == *this);
    }
    //Implements pprint of a constant
    // Set 'opencl' for OpenCL specific output
    void pprint(std::ostream& out, bool opencl=false) const;
};

//Implements pprint of a constant (by streaming)
std::ostream& operator<<(std::ostream& out, const bh_constant& constant);
