/*
This file is part of Bohrium and copyright (c) 2012 the Bohrium team:
http://bohrium.bitbucket.org

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

//
//  WARN:   This file is generated; changes to it will be overwritten.
//          If you wish to change its functionality then change the code-generator for this file.
//          Take a look at: codegen/README
//


#ifndef __BOHRIUM_BRIDGE_CPP_TRAITS
#define __BOHRIUM_BRIDGE_CPP_TRAITS
#include <complex>

namespace bxx {
    template <typename T>
inline
void assign_const_type(bh_constant* constant, T value) {
    //TODO: The general case should result in a meaningful compile-time error.
    std::cout << "Unsupported type [%s, " << bh_type_text(constant->type) << "] " << &value << std::endl;
}

    template <>
    inline
    void assign_const_type(bh_constant* constant, bool value)
    {
        constant->value.bool8 = value;
        constant->type = bh_type::BOOL;
    }

    template <>
    inline
    void assign_const_type(bh_constant* constant, int8_t value)
    {
        constant->value.int8 = value;
        constant->type = bh_type::INT8;
    }

    template <>
    inline
    void assign_const_type(bh_constant* constant, int16_t value)
    {
        constant->value.int16 = value;
        constant->type = bh_type::INT16;
    }

    template <>
    inline
    void assign_const_type(bh_constant* constant, int32_t value)
    {
        constant->value.int32 = value;
        constant->type = bh_type::INT32;
    }

    template <>
    inline
    void assign_const_type(bh_constant* constant, int64_t value)
    {
        constant->value.int64 = value;
        constant->type = bh_type::INT64;
    }

    template <>
    inline
    void assign_const_type(bh_constant* constant, uint8_t value)
    {
        constant->value.uint8 = value;
        constant->type = bh_type::UINT8;
    }

    template <>
    inline
    void assign_const_type(bh_constant* constant, uint16_t value)
    {
        constant->value.uint16 = value;
        constant->type = bh_type::UINT16;
    }

    template <>
    inline
    void assign_const_type(bh_constant* constant, uint32_t value)
    {
        constant->value.uint32 = value;
        constant->type = bh_type::UINT32;
    }

    template <>
    inline
    void assign_const_type(bh_constant* constant, uint64_t value)
    {
        constant->value.uint64 = value;
        constant->type = bh_type::UINT64;
    }

    template <>
    inline
    void assign_const_type(bh_constant* constant, float value)
    {
        constant->value.float32 = value;
        constant->type = bh_type::FLOAT32;
    }

    template <>
    inline
    void assign_const_type(bh_constant* constant, double value)
    {
        constant->value.float64 = value;
        constant->type = bh_type::FLOAT64;
    }


template <>
inline
void assign_const_type(bh_constant* constant, bh_complex64 value)
{
    constant->value.complex64 = value;
    constant->type = bh_type::COMPLEX64;
}

template <>
inline
void assign_const_type(bh_constant* constant, bh_complex128 value)
{
    constant->value.complex128 = value;
    constant->type = bh_type::COMPLEX128;
}

template <>
inline
void assign_const_type(bh_constant* constant, std::complex<float> value)
{
    constant->value.complex64.real = value.real();
    constant->value.complex64.imag = value.imag();
    constant->type = bh_type::COMPLEX64;
}

template <>
inline
void assign_const_type(bh_constant* constant, std::complex<double> value)
{
    constant->value.complex128.real = value.real();
    constant->value.complex128.imag = value.imag();
    constant->type = bh_type::COMPLEX128;
}

    template <typename T>
inline
void assign_array_type(bh_base* base) {
    // TODO: The general case should result in a meaningful compile-time error.
    std::cout << "Unsupported type: " << base << std::endl;
}

template <>
inline
void assign_array_type<bool>(bh_base* base)
{
    base->type = bh_type::BOOL;
}

template <>
inline
void assign_array_type<int8_t>(bh_base* base)
{
    base->type = bh_type::INT8;
}

template <>
inline
void assign_array_type<int16_t>(bh_base* base)
{
    base->type = bh_type::INT16;
}

template <>
inline
void assign_array_type<int32_t>(bh_base* base)
{
    base->type = bh_type::INT32;
}

template <>
inline
void assign_array_type<int64_t>(bh_base* base)
{
    base->type = bh_type::INT64;
}

template <>
inline
void assign_array_type<uint8_t>(bh_base* base)
{
    base->type = bh_type::UINT8;
}

template <>
inline
void assign_array_type<uint16_t>(bh_base* base)
{
    base->type = bh_type::UINT16;
}

template <>
inline
void assign_array_type<uint32_t>(bh_base* base)
{
    base->type = bh_type::UINT32;
}

template <>
inline
void assign_array_type<uint64_t>(bh_base* base)
{
    base->type = bh_type::UINT64;
}

template <>
inline
void assign_array_type<float>(bh_base* base)
{
    base->type = bh_type::FLOAT32;
}

template <>
inline
void assign_array_type<double>(bh_base* base)
{
    base->type = bh_type::FLOAT64;
}

template <>
inline
void assign_array_type<std::complex<float> >(bh_base* base)
{
    base->type = bh_type::COMPLEX64;
}

template <>
inline
void assign_array_type<std::complex<double> >(bh_base* base)
{
    base->type = bh_type::COMPLEX128;
}


template <>
inline
void assign_array_type<bh_complex64>(bh_base* base)
{
    base->type = bh_type::COMPLEX64;
}

template <>
inline
void assign_array_type<bh_complex128>(bh_base* base)
{
    base->type = bh_type::COMPLEX128;
}

}

#endif
