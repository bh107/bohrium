/*
This file is part of cphVB and copyright (c) 2012 the cphVB team:
http://cphvb.bitbucket.org

cphVB is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as 
published by the Free Software Foundation, either version 3 
of the License, or (at your option) any later version.

cphVB is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the 
GNU Lesser General Public License along with cphVB. 

If not, see <http://www.gnu.org/licenses/>.
*/
#ifndef __BOHRIUM_BRIDGE_CPP_TRAITS
#define __BOHRIUM_BRIDGE_CPP_TRAITS
#include "cphvb.h"

namespace bh {

template <typename T>
inline
void assign_array_type( cphvb_array* array );
// NOTE: The general implementation could output an error at runtime instead of failing at compile-time.

template <>
inline
void assign_array_type<cphvb_bool>( cphvb_array* array )
{
    array->type = CPHVB_BOOL;
}

template <>
inline
void assign_array_type<cphvb_int8>( cphvb_array* array )
{
    array->type = CPHVB_INT8;
}

template <>
inline
void assign_array_type<cphvb_int16>( cphvb_array* array )
{
    array->type = CPHVB_INT16;
}

template <>
inline
void assign_array_type<cphvb_int32>( cphvb_array* array )
{
    array->type = CPHVB_INT32;
}

template <>
inline
void assign_array_type<cphvb_int64>( cphvb_array* array )
{
    array->type = CPHVB_INT64;
}

template <>
inline
void assign_array_type<cphvb_uint8>( cphvb_array* array )
{
    array->type = CPHVB_UINT8;
}

template <>
inline
void assign_array_type<cphvb_uint16>( cphvb_array* array )
{
    array->type = CPHVB_UINT16;
}

template <>
inline
void assign_array_type<cphvb_uint32>( cphvb_array* array )
{
    array->type = CPHVB_UINT32;
}

template <>
inline
void assign_array_type<cphvb_uint64>( cphvb_array* array )
{
    array->type = CPHVB_UINT64;
}

template <>
inline
void assign_array_type<cphvb_float16>( cphvb_array* array )
{
    array->type = CPHVB_FLOAT16;
}

template <>
inline
void assign_array_type<cphvb_float32>( cphvb_array* array )
{
    array->type = CPHVB_FLOAT32;
}

template <>
inline
void assign_array_type<cphvb_float64>( cphvb_array* array )
{
    array->type = CPHVB_FLOAT64;
}

template <>
inline
void assign_array_type<cphvb_complex64>( cphvb_array* array )
{
    array->type = CPHVB_COMPLEX64;
}

template <>
inline
void assign_array_type<cphvb_complex128>( cphvb_array* array )
{
    array->type = CPHVB_COMPLEX128;
}

template <typename T>
inline
void assign_const_type( cphvb_constant* instr, T value );
// NOTE: The general implementation could output an error at runtime instead of failing at compile-time.

template <>
inline
void assign_const_type( cphvb_constant* constant, cphvb_bool value )
{
    constant->value.bool8 = value;
    constant->type = CPHVB_BOOL;
}

template <>
inline
void assign_const_type( cphvb_constant* constant, cphvb_int8 value )
{
    constant->value.int8 = value;
    constant->type = CPHVB_INT8;
}

template <>
inline
void assign_const_type( cphvb_constant* constant, cphvb_int16 value )
{
    constant->value.int16 = value;
    constant->type = CPHVB_INT16;
}

template <>
inline
void assign_const_type( cphvb_constant* constant, cphvb_int32 value )
{
    constant->value.int32 = value;
    constant->type = CPHVB_INT32;
}

template <>
inline
void assign_const_type( cphvb_constant* constant, cphvb_int64 value )
{
    constant->value.int64 = value;
    constant->type = CPHVB_INT64;
}

template <>
inline
void assign_const_type( cphvb_constant* constant, cphvb_uint8 value )
{
    constant->value.uint8 = value;
    constant->type = CPHVB_UINT8;
}

template <>
inline
void assign_const_type( cphvb_constant* constant, cphvb_uint16 value )
{
    constant->value.uint16 = value;
    constant->type = CPHVB_UINT16;
}

template <>
inline
void assign_const_type( cphvb_constant* constant, cphvb_uint32 value )
{
    constant->value.uint32 = value;
    constant->type = CPHVB_UINT32;
}

template <>
inline
void assign_const_type( cphvb_constant* constant, cphvb_uint64 value )
{
    constant->value.uint64 = value;
    constant->type = CPHVB_UINT64;
}

template <>
inline
void assign_const_type( cphvb_constant* constant, cphvb_float16 value )
{
    constant->value.float16 = value;
    constant->type = CPHVB_FLOAT16;
}

template <>
inline
void assign_const_type( cphvb_constant* constant, cphvb_float32 value )
{
    constant->value.float32 = value;
    constant->type = CPHVB_FLOAT32;
}

template <>
inline
void assign_const_type( cphvb_constant* constant, cphvb_float64 value )
{
    constant->value.float64 = value;
    constant->type = CPHVB_FLOAT64;
}

template <>
inline
void assign_const_type( cphvb_constant* constant, cphvb_complex64 value )
{
    constant->value.complex64 = value;
    constant->type = CPHVB_COMPLEX64;
}

template <>
inline
void assign_const_type( cphvb_constant* constant, cphvb_complex128 value )
{
    constant->value.complex128 = value;
    constant->type = CPHVB_COMPLEX128;
}

}

#endif
