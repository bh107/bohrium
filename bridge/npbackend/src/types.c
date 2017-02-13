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

#include "types.h"
#include <numpy/npy_math.h>

void bh_set_constant(int npy_type, bh_constant* constant, void* data) {
    constant->type = type_py2cph(npy_type);

    switch (npy_type) {
        case NPY_BOOL:
            constant->value.bool8 = *(npy_bool*)data;
        case NPY_BYTE:
            constant->value.int8 = *(npy_byte*)data;
        case NPY_UBYTE:
            constant->value.uint8 = *(npy_ubyte*)data;
        case NPY_SHORT:
            constant->value.int16 = *(npy_short*)data;
        case NPY_USHORT:
            constant->value.uint16 = *(npy_ushort*)data;
    #if NPY_BITSOF_LONG == 32
        case NPY_LONG:
    #endif
        case NPY_INT:
            constant->value.int32 = *(npy_int*)data;
    #if NPY_BITSOF_LONG == 32
        case NPY_ULONG:
    #endif
        case NPY_UINT:
            constant->value.uint32 = *(npy_uint*)data;
    #if NPY_BITSOF_LONG == 64
        case NPY_LONG:
    #endif
        case NPY_LONGLONG:
            constant->value.int64 = *(npy_longlong*)data;
    #if NPY_BITSOF_LONG == 64
        case NPY_ULONG:
    #endif
        case NPY_ULONGLONG:
            constant->value.uint64 = *(npy_ulonglong*)data;
        case NPY_FLOAT:
            constant->value.float32 = *(npy_float*)data;
        case NPY_DOUBLE:
            constant->value.float64 = *(npy_double*)data;
        case NPY_COMPLEX64:
            constant->value.complex64.real = npy_crealf(*(npy_cfloat*)data);
            constant->value.complex64.imag = npy_cimagf(*(npy_cfloat*)data);
        case NPY_COMPLEX128:
            constant->value.complex128.real = npy_creal(*(npy_cdouble*)data);
            constant->value.complex128.imag = npy_cimag(*(npy_cdouble*)data);
        default:
            throw std::runtime_error("Unknown type for bh_set_constant()");
    }
}

void bh_set_int_constant(int npy_type, bh_constant* constant, long long integer)
{
    constant->type = type_py2cph(npy_type);
    switch (npy_type) {
        case NPY_BOOL:
            constant->value.bool8 = integer;
        case NPY_BYTE:
            constant->value.int8 = integer;
        case NPY_UBYTE:
            constant->value.uint8 = integer;
        case NPY_SHORT:
            constant->value.int16 = integer;
        case NPY_USHORT:
            constant->value.uint16 = integer;
    #if NPY_BITSOF_LONG == 32
        case NPY_LONG:
    #endif
        case NPY_INT:
            constant->value.int32 = integer;
    #if NPY_BITSOF_LONG == 32
        case NPY_ULONG:
    #endif
        case NPY_UINT:
            constant->value.uint32 = integer;
    #if NPY_BITSOF_LONG == 64
        case NPY_LONG:
    #endif
        case NPY_LONGLONG:
            constant->value.int64 = integer;
    #if NPY_BITSOF_LONG == 64
        case NPY_ULONG:
    #endif
        case NPY_ULONGLONG:
            constant->value.uint64 = integer;
        case NPY_FLOAT:
            constant->value.float32 = integer;
        case NPY_DOUBLE:
            constant->value.float64 = integer;
        case NPY_COMPLEX64:
            constant->value.complex64.real = integer;
            constant->value.complex64.imag = 0;
        case NPY_COMPLEX128:
            constant->value.complex128.real = integer;
            constant->value.complex128.imag = 0;
        default:
            throw std::runtime_error("Unknown type for bh_set_int_constant()");
    }
}


/*===================================================================
 *
 * The data type conversion to and from NumPy and Bohrium.
 * Private.
 */
bh_type type_py2cph(int npy_type)
{
    switch(npy_type)
    {
        case NPY_BOOL:      return BH_BOOL;
        case NPY_BYTE:      return BH_INT8;
        case NPY_UBYTE:     return BH_UINT8;
        case NPY_SHORT:     return BH_INT16;
        case NPY_USHORT:    return BH_UINT16;
        case NPY_INT:       return BH_INT32;
        case NPY_UINT:      return BH_UINT32;
        #if NPY_BITSOF_LONG == 32
            case NPY_LONG:  return BH_INT32;
            case NPY_ULONG: return BH_UINT32;
        #else
            case NPY_LONG:  return BH_INT64;
            case NPY_ULONG: return BH_UINT64;
        #endif
        case NPY_LONGLONG:  return BH_INT64;
        case NPY_ULONGLONG: return BH_UINT64;
        case NPY_FLOAT:     return BH_FLOAT32;
        case NPY_DOUBLE:    return BH_FLOAT64;
        case NPY_CFLOAT:    return BH_COMPLEX64;
        case NPY_CDOUBLE:   return BH_COMPLEX128;
        default:            return BH_UNKNOWN;
    }
};
