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

/** A Bohrium constant */
class bh_constant {
public:
    // The value of the constant
    bh_constant_value value;
    // The data type of the constant
    bh_type type;

    /** Set the value of this constant.
     * The `value` is static_cast'ed to the type of the constant
     *
     * @tparam T    The type of `value` (NB: the type of the constant isn't affected)
     * @param value The new value
     */
    template<typename T>
    void set_value(T value) {
        switch(type) {
            case bh_type::BOOL:
                this->value = bh_constant_value{static_cast<bool>(value)};
                return;
            case bh_type::INT8:
                this->value = bh_constant_value{static_cast<int8_t>(value)};
                return;
            case bh_type::INT16:
                this->value = bh_constant_value{static_cast<int16_t>(value)};
                return;
            case bh_type::INT32:
                this->value = bh_constant_value{static_cast<int32_t>(value)};
                return;
            case bh_type::INT64:
                this->value = bh_constant_value{static_cast<int64_t>(value)};
                return;
            case bh_type::UINT8:
                this->value = bh_constant_value{static_cast<uint8_t>(value)};
                return;
            case bh_type::UINT16:
                this->value = bh_constant_value{static_cast<uint16_t>(value)};
                return;
            case bh_type::UINT32:
                this->value = bh_constant_value{static_cast<uint32_t>(value)};
                return;
            case bh_type::UINT64:
                this->value = bh_constant_value{static_cast<uint64_t>(value)};
                return;
            case bh_type::FLOAT32:
                this->value = bh_constant_value{static_cast<float>(value)};
                return;
            case bh_type::FLOAT64:
                this->value = bh_constant_value{static_cast<double>(value)};
                return;
            case bh_type::COMPLEX64:
                this->value = bh_constant_value{static_cast<std::complex<float> >(value)};
                return;
            case bh_type::COMPLEX128:
                this->value = bh_constant_value{static_cast<std::complex<double> >(value)};
                return;
            case bh_type::R123:
                this->value = bh_constant_value{static_cast<uint64_t>(value)};
                return;
            default:
                throw std::runtime_error("set_value(): unknown constant type");
        }
    }

    /** `bh_123` specialization of `set_value()` */
    void set_value(bh_r123 value) {
        if (type == bh_type::R123) {
            this->value = bh_constant_value{value};
        } else {
            throw std::runtime_error("set_value(): can only set BH_R123 constants with type `bh_r123`");
        }
    }

    /** Default constructor */
    bh_constant() = default;

    /** Constructor where the type of `val` and the new constant are the same
     *
     * @tparam T  The type of `val` and the new constant
     * @param val The value of the new constant
     */
    template<typename T>
    bh_constant(T val) : value(val), type(bh_type_from_template<T>()) {}

    /** Constructor where the type of `val` and the type of the new constant might not be the same
     *
     * @tparam T   The type of `val`
     * @param val  The value of the new constant
     * @param type The type of the new constant
     */
    template<typename T>
    bh_constant(T val, bh_type type) : type(type) {
        set_value(val);
    }

    /** Factory method that creates a new constant with the minimum value possible
     *
     * @param type The type of the new constant
     * @return     The new constant
     */
    static bh_constant get_min(bh_type type);

    /** Factory method that creates a new constant with the maximum value possible
     *
     * @param type The type of the new constant
     * @return     The new constant
     */
    static bh_constant get_max(bh_type type);

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

    bool operator!=(const bh_constant& other) const {
        return !(other == *this);
    }

    //Implements pprint of a constant
    // Set 'opencl' for OpenCL specific output
    void pprint(std::ostream& out, bool opencl=false) const;
};

//Implements pprint of a constant (by streaming)
std::ostream& operator<<(std::ostream& out, const bh_constant& constant);
