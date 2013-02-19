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
#ifndef __BOHRIUM_BRIDGE_CPP_TRAITS
#define __BOHRIUM_BRIDGE_CPP_TRAITS
#include "bh.h"

namespace bh {

template <typename T>
inline
void assign_const_type( bh_constant* instr, T value );
// NOTE: The general implementation could output an error at runtime instead of failing at compile-time.

template <>
inline
void assign_const_type( bh_constant* constant, bh_bool value )
{
    constant->value.bool8 = value;
    constant->type = BH_BOOL;
}

template <>
inline
void assign_const_type( bh_constant* constant, bh_int8 value )
{
    constant->value.int8 = value;
    constant->type = BH_INT8;
}

template <>
inline
void assign_const_type( bh_constant* constant, bh_int16 value )
{
    constant->value.int16 = value;
    constant->type = BH_INT16;
}

template <>
inline
void assign_const_type( bh_constant* constant, bh_int32 value )
{
    constant->value.int32 = value;
    constant->type = BH_INT32;
}

template <>
inline
void assign_const_type( bh_constant* constant, bh_int64 value )
{
    constant->value.int64 = value;
    constant->type = BH_INT64;
}

template <>
inline
void assign_const_type( bh_constant* constant, bh_uint16 value )
{
    constant->value.uint16 = value;
    constant->type = BH_UINT16;
}

template <>
inline
void assign_const_type( bh_constant* constant, bh_uint32 value )
{
    constant->value.uint32 = value;
    constant->type = BH_UINT32;
}

template <>
inline
void assign_const_type( bh_constant* constant, bh_uint64 value )
{
    constant->value.uint64 = value;
    constant->type = BH_UINT64;
}

template <>
inline
void assign_const_type( bh_constant* constant, bh_float32 value )
{
    constant->value.float32 = value;
    constant->type = BH_FLOAT32;
}

template <>
inline
void assign_const_type( bh_constant* constant, bh_float64 value )
{
    constant->value.float64 = value;
    constant->type = BH_FLOAT64;
}

template <>
inline
void assign_const_type( bh_constant* constant, bh_complex64 value )
{
    constant->value.complex64 = value;
    constant->type = BH_COMPLEX64;
}

template <>
inline
void assign_const_type( bh_constant* constant, bh_complex128 value )
{
    constant->value.complex128 = value;
    constant->type = BH_COMPLEX128;
}

template <typename T>
inline
void assign_array_type( bh_array* array );
// NOTE: The general implementation could output an error at runtime instead of failing at compile-time.

template <>
inline
void assign_array_type<bh_bool>( bh_array* array )
{
    array->type = BH_BOOL;
}

template <>
inline
void assign_array_type<bh_int8>( bh_array* array )
{
    array->type = BH_INT8;
}

template <>
inline
void assign_array_type<bh_int16>( bh_array* array )
{
    array->type = BH_INT16;
}

template <>
inline
void assign_array_type<bh_int32>( bh_array* array )
{
    array->type = BH_INT32;
}

template <>
inline
void assign_array_type<bh_int64>( bh_array* array )
{
    array->type = BH_INT64;
}

template <>
inline
void assign_array_type<bh_uint16>( bh_array* array )
{
    array->type = BH_UINT16;
}

template <>
inline
void assign_array_type<bh_uint32>( bh_array* array )
{
    array->type = BH_UINT32;
}

template <>
inline
void assign_array_type<bh_uint64>( bh_array* array )
{
    array->type = BH_UINT64;
}

template <>
inline
void assign_array_type<bh_float32>( bh_array* array )
{
    array->type = BH_FLOAT32;
}

template <>
inline
void assign_array_type<bh_float64>( bh_array* array )
{
    array->type = BH_FLOAT64;
}

template <>
inline
void assign_array_type<bh_complex64>( bh_array* array )
{
    array->type = BH_COMPLEX64;
}

template <>
inline
void assign_array_type<bh_complex128>( bh_array* array )
{
    array->type = BH_COMPLEX128;
}


}
