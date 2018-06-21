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

#include <stdexcept>
#include <limits>
#include <iostream>
#include <complex>

#include <bh_constant.hpp>
#include <bh_type.hpp>
#include <cmath>

using namespace std;

bh_constant bh_constant::get_min(bh_type type) {
    switch (type) {
        case bh_type::BOOL:
            return bh_constant(bh_bool{0});
        case bh_type::INT8:
            return bh_constant(std::numeric_limits<int8_t>::min() + 1); // Notice, +1 is to make sure no overflow
        case bh_type::INT16:
            return bh_constant(std::numeric_limits<int16_t>::min() + 1);
        case bh_type::INT32:
            return bh_constant(std::numeric_limits<int32_t>::min() + 1);
        case bh_type::INT64:
            return bh_constant(std::numeric_limits<int64_t>::min() + 1);
        case bh_type::UINT8:
            return bh_constant(bh_uint8{0});
        case bh_type::UINT16:
            return bh_constant(bh_uint16{0});
        case bh_type::UINT32:
            return bh_constant(bh_uint32{0});
        case bh_type::UINT64:
            return bh_constant(bh_uint64{0});
        case bh_type::FLOAT32:
            return bh_constant(std::numeric_limits<float>::min());
        case bh_type::FLOAT64:
            return bh_constant(std::numeric_limits<double>::min());
        case bh_type::COMPLEX64:
            return bh_constant(std::complex<float>(std::numeric_limits<float>::min(),
                                                   std::numeric_limits<float>::min()));
        case bh_type::COMPLEX128:
            return bh_constant(std::complex<double>(std::numeric_limits<double>::min(),
                                                    std::numeric_limits<double>::min()));
        case bh_type::R123:
            return bh_constant(bh_r123{0, 0});
        default:
            throw runtime_error("bh_constant::get_min(): unknown type");
    }
}

bh_constant bh_constant::get_max(bh_type type) {
    switch (type) {
        case bh_type::BOOL:
            return bh_constant(bh_bool{1});
        case bh_type::INT8:
            return bh_constant(std::numeric_limits<int8_t>::max());
        case bh_type::INT16:
            return bh_constant(std::numeric_limits<int16_t>::max());
        case bh_type::INT32:
            return bh_constant(std::numeric_limits<int32_t>::max());
        case bh_type::INT64:
            return bh_constant(std::numeric_limits<int64_t>::max());
        case bh_type::UINT8:
            return bh_constant(std::numeric_limits<uint8_t>::max());
        case bh_type::UINT16:
            return bh_constant(std::numeric_limits<uint16_t>::max());
        case bh_type::UINT32:
            return bh_constant(std::numeric_limits<uint32_t>::max());
        case bh_type::UINT64:
            return bh_constant(std::numeric_limits<uint64_t>::max());
        case bh_type::FLOAT32:
            return bh_constant(std::numeric_limits<float>::max());
        case bh_type::FLOAT64:
            return bh_constant(std::numeric_limits<double>::max());
        case bh_type::COMPLEX64:
            return bh_constant(std::complex<float>(std::numeric_limits<float>::max(),
                                                   std::numeric_limits<float>::max()));
        case bh_type::COMPLEX128:
            return bh_constant(std::complex<double>(std::numeric_limits<double>::max(),
                                                    std::numeric_limits<double>::max()));
        case bh_type::R123:
            return bh_constant(bh_r123{std::numeric_limits<uint64_t>::max(), std::numeric_limits<uint64_t>::max()});
        default:
            throw runtime_error("bh_constant::get_min(): unknown type");
    }
}


int64_t bh_constant::get_int64() const
{
    switch(type) {
        case bh_type::BOOL:
            return int64_t{value.bool8};
        case bh_type::INT8:
            return int64_t{value.int8};
        case bh_type::INT16:
            return int64_t{value.int16};
        case bh_type::INT32:
            return int64_t{value.int32};
        case bh_type::INT64:
            return value.int64;
        case bh_type::UINT8:
            return int64_t{value.uint8};
        case bh_type::UINT16:
            return int64_t{value.uint16};
        case bh_type::UINT32:
            return int64_t{value.uint32};
        case bh_type::UINT64:
            // We have to check that 'value' isn't too big for an int64_t
            if (value.uint64 < numeric_limits<int64_t>::max())
                return static_cast<int64_t>(value.uint64);
        default:
            throw overflow_error("Constant cannot be converted to int64_t");
    }
}

uint64_t bh_constant::get_uint64() const
{
    switch(type) {
        case bh_type::UINT8:
            return uint64_t{value.uint8};
        case bh_type::UINT16:
            return uint64_t{value.uint16};
        case bh_type::UINT32:
            return uint64_t{value.uint32};
        case bh_type::UINT64:
            return value.uint64;
        default:
            throw overflow_error("Constant cannot be converted to uint64_t");
    }
}

double bh_constant::get_double() const
{
    switch(type) {
        case bh_type::BOOL:
            return static_cast<double>(value.bool8);
        case bh_type::INT8:
            return static_cast<double>(value.int8);
        case bh_type::INT16:
            return static_cast<double>(value.int16);
        case bh_type::INT32:
            return static_cast<double>(value.int32);
        case bh_type::INT64:
            return static_cast<double>(value.int64);
        case bh_type::UINT8:
            return static_cast<double>(value.uint8);
        case bh_type::UINT16:
            return static_cast<double>(value.uint16);
        case bh_type::UINT32:
            return static_cast<double>(value.uint32);
        case bh_type::UINT64:
            return static_cast<double>(value.uint64);
        case bh_type::FLOAT32:
            return static_cast<double>(value.float32);
        case bh_type::FLOAT64:
            return value.float64;
        case bh_type::COMPLEX64:
            if (value.complex64.imag != 0){
                throw overflow_error("Complex64 cannot be converted"
                                     "to double when imag isn't zero");
            }
            return static_cast<double>(value.complex64.real);
        case bh_type::COMPLEX128:
            if (value.complex128.imag != 0){
                throw overflow_error("Complex128 cannot be converted"
                                     "to double when imag isn't zero");
            }
            return static_cast<double>(value.complex128.real);
        case bh_type::R123:
            throw overflow_error("R123 cannot be converted to double");
        default:
            throw runtime_error("Unknown constant type in get_double");
    }
}

void bh_constant::set_double(double value)
{
    switch(type) {
        case bh_type::BOOL:
            this->value.bool8 = static_cast<bool>(value);
            return;
        case bh_type::INT8:
            this->value.int8 = static_cast<int8_t>(value);
            return;
        case bh_type::INT16:
            this->value.int16 = static_cast<int16_t>(value);
            return;
        case bh_type::INT32:
            this->value.int32 = static_cast<int32_t>(value);
            return;
        case bh_type::INT64:
            this->value.int64 = static_cast<int64_t>(value);
            return;
        case bh_type::UINT8:
            this->value.uint8 = static_cast<uint8_t>(value);
            return;
        case bh_type::UINT16:
            this->value.uint16 = static_cast<uint16_t>(value);
            return;
        case bh_type::UINT32:
            this->value.uint32 = static_cast<uint32_t>(value);
            return;
        case bh_type::UINT64:
            this->value.uint64 = static_cast<uint64_t>(value);
            return;
        case bh_type::FLOAT32:
            this->value.float32 = static_cast<float>(value);
            return;
        case bh_type::FLOAT64:
            this->value.float64 = value;
            return;
        case bh_type::COMPLEX64:
            this->value.complex64.real = static_cast<int32_t>(value);
            this->value.complex64.imag = 0;
            return;
        case bh_type::COMPLEX128:
            this->value.complex128.real = static_cast<int64_t>(value);
            this->value.complex128.imag = 0;
            return;
        case bh_type::R123:
            throw overflow_error("double to R123 isn't possible");
        default:
            throw runtime_error("Unknown constant type in set_double");
    }
}

bool bh_constant::operator==(const bh_constant& other) const
{
    if (other.type != type) return false;

    switch (type) {
        case bh_type::BOOL:
            return other.value.bool8 == value.bool8;
        case bh_type::INT8:
            return other.value.int8 == value.int8;
        case bh_type::INT16:
            return other.value.int16 == value.int16;
        case bh_type::INT32:
            return other.value.int32 == value.int32;
        case bh_type::INT64:
            return other.value.int64 == value.int64;
        case bh_type::UINT8:
            return other.value.uint8 == value.uint8;
        case bh_type::UINT16:
            return other.value.uint16 == value.uint16;
        case bh_type::UINT32:
            return other.value.uint32 == value.uint32;
        case bh_type::UINT64:
            return other.value.uint64 == value.uint64;
        case bh_type::FLOAT32:
            return other.value.float32 == value.float32;
        case bh_type::FLOAT64:
            return other.value.float64 == value.float64;
        case bh_type::COMPLEX64:
            return other.value.complex64.real == value.complex64.real &&
                   other.value.complex64.imag == value.complex64.imag;
        case bh_type::COMPLEX128:
            return other.value.complex128.real == value.complex128.real &&
                   other.value.complex128.imag == value.complex128.imag;
        case bh_type::R123:
            return other.value.r123.start == value.r123.start &&
                   other.value.r123.key == value.r123.key;
        default:
            return false;
    }
}

namespace {
    // Print float while handling nan and inf
    void ppfloat(float value, ostream& out) {
        if (value != value) {
            out << "NAN";
        } else if (std::isinf(value)) {
            if (signbit(value)) {
                out << "(-INFINITY)";
            } else {
                out << "INFINITY";
            }
        } else {
            out << value << "f";
        }
    }
    void ppfloat(double value, ostream& out) {
        if (value != value) {
            out << "NAN";
        } else if (std::isinf(value)) {
            if (signbit(value)) {
                out << "(-INFINITY)";
            } else {
                out << "INFINITY";
            }
        } else {
            out << value;
        }
    }
}

void bh_constant::pprint(ostream& out, bool opencl) const
{
    if (type == bh_type::BOOL) {
        out << get_int64();
    } else if (bh_type_is_integer(type)) {
        if (bh_type_is_signed_integer(type)) {
            out << get_int64();
        } else {
            out << get_uint64() << "u";
        }
    } else {
        out.precision(numeric_limits<double>::max_digits10);
        out << std::scientific;
        switch(type) {
            case bh_type::FLOAT32:
                ppfloat(value.float32, out);
                break;
            case bh_type::FLOAT64:
                ppfloat(value.float64, out);
                break;
            case bh_type::R123:
                out << "{.start = " << value.r123.start << ", .key = " << value.r123.key << "}";
                break;
            case bh_type::COMPLEX64:
                if (opencl) {
                    out << "make_complex64(";
                    ppfloat(value.complex64.real, out);
                    out << ", ";
                    ppfloat(value.complex64.imag, out);
                    out << ")";
                } else {
                    out << "(";
                    ppfloat(value.complex64.real, out);
                    out << " + ";
                    ppfloat(value.complex64.imag, out);
                    out << "*I)";
                }
                break;
            case bh_type::COMPLEX128:
                if (opencl) {
                    out << "make_complex128(";
                    ppfloat(value.complex128.real, out);
                    out << ", ";
                    ppfloat(value.complex128.imag, out);
                    out << ")";
                } else {
                    out << "(";
                    ppfloat(value.complex128.real, out);
                    out << " + ";
                    ppfloat(value.complex128.imag, out);
                    out << "*I)";
                }
                break;
            default:
                out << "?";
        }
        out.unsetf(std::ios_base::floatfield); // Resetting the float formatting
    }
}

ostream& operator<<(ostream& out, const bh_constant& constant)
{
    constant.pprint(out);
    return out;
}
