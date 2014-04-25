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

bh_error bh_set_constant(int npy_type, bh_constant* constant,
                         void* data)
{
    constant->type = type_py2cph(npy_type);
    switch (npy_type)
    {
    case NPY_BOOL:
        constant->value.bool8 = *(npy_bool*)data;
        return BH_SUCCESS;
    case NPY_BYTE:
        constant->value.int8 = *(npy_byte*)data;
        return BH_SUCCESS;
    case NPY_UBYTE:
        constant->value.uint8 = *(npy_ubyte*)data;
        return BH_SUCCESS;
    case NPY_SHORT:
        constant->value.int16 = *(npy_short*)data;
        return BH_SUCCESS;
    case NPY_USHORT:
        constant->value.uint16 = *(npy_ushort*)data;
        return BH_SUCCESS;
#if NPY_BITSOF_LONG == 32
    case NPY_LONG:
#endif
    case NPY_INT:
        constant->value.int32 = *(npy_int*)data;
        return BH_SUCCESS;
#if NPY_BITSOF_LONG == 32
    case NPY_ULONG:
#endif
    case NPY_UINT:
        constant->value.uint32 = *(npy_uint*)data;
        return BH_SUCCESS;
#if NPY_BITSOF_LONG == 64
    case NPY_LONG:
#endif
    case NPY_LONGLONG:
        constant->value.int64 = *(npy_longlong*)data;
        return BH_SUCCESS;
#if NPY_BITSOF_LONG == 64
    case NPY_ULONG:
#endif
    case NPY_ULONGLONG:
        constant->value.uint64 = *(npy_ulonglong*)data;
        return BH_SUCCESS;
    case NPY_FLOAT:
        constant->value.float32 = *(npy_float*)data;
        return BH_SUCCESS;
    case NPY_DOUBLE:
        constant->value.float64 = *(npy_double*)data;
        return BH_SUCCESS;
    case NPY_COMPLEX64:
        constant->value.complex64.real = npy_crealf(*(npy_cfloat*)data);
        constant->value.complex64.imag = npy_cimagf(*(npy_cfloat*)data);
        return BH_SUCCESS;
    case NPY_COMPLEX128:
        constant->value.complex128.real = npy_creal(*(npy_cdouble*)data);
        constant->value.complex128.imag = npy_cimag(*(npy_cdouble*)data);
        return BH_SUCCESS;
    default:
        return BH_ERROR;

    }
}

bh_error bh_set_int_constant(int npy_type, bh_constant* constant, long long integer)
{
    constant->type = type_py2cph(npy_type);
    switch (npy_type)
    {
    case NPY_BOOL:
        constant->value.bool8 = integer;
        return BH_SUCCESS;
    case NPY_BYTE:
        constant->value.int8 = integer;
        return BH_SUCCESS;
    case NPY_UBYTE:
        constant->value.uint8 = integer;
        return BH_SUCCESS;
    case NPY_SHORT:
        constant->value.int16 = integer;
        return BH_SUCCESS;
    case NPY_USHORT:
        constant->value.uint16 = integer;
        return BH_SUCCESS;
#if NPY_BITSOF_LONG == 32
    case NPY_LONG:
#endif
    case NPY_INT:
        constant->value.int32 = integer;
        return BH_SUCCESS;
#if NPY_BITSOF_LONG == 32
    case NPY_ULONG:
#endif
    case NPY_UINT:
        constant->value.uint32 = integer;
        return BH_SUCCESS;
#if NPY_BITSOF_LONG == 64
    case NPY_LONG:
#endif
    case NPY_LONGLONG:
        constant->value.int64 = integer;
        return BH_SUCCESS;
#if NPY_BITSOF_LONG == 64
    case NPY_ULONG:
#endif
    case NPY_ULONGLONG:
        constant->value.uint64 = integer;
        return BH_SUCCESS;
    case NPY_FLOAT:
        constant->value.float32 = integer;
        return BH_SUCCESS;
    case NPY_DOUBLE:
        constant->value.float64 = integer;
        return BH_SUCCESS;
    case NPY_COMPLEX64:
        constant->value.complex64.real = integer;
        constant->value.complex64.imag = 0;
        return BH_SUCCESS;
    case NPY_COMPLEX128:
        constant->value.complex128.real = integer;
        constant->value.complex128.imag = 0;
        return BH_SUCCESS;
    default:
		assert(23 == 43);
        return BH_ERROR;

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
        case NPY_BOOL: return   BH_BOOL;
        case NPY_BYTE: return   BH_INT8;
        case NPY_UBYTE: return  BH_UINT8;
        case NPY_SHORT: return  BH_INT16;
        case NPY_USHORT: return BH_UINT16;
        case NPY_INT: return    BH_INT32;
        case NPY_UINT: return   BH_UINT32;
        #if NPY_BITSOF_LONG == 32
            case NPY_LONG: return  BH_INT32;
            case NPY_ULONG: return BH_UINT32;
        #else
            case NPY_LONG: return  BH_INT64;
            case NPY_ULONG: return BH_UINT64;
        #endif
        case NPY_LONGLONG: return    BH_INT64;
        case NPY_ULONGLONG: return   BH_UINT64;
        case NPY_FLOAT: return       BH_FLOAT32;
        case NPY_DOUBLE: return      BH_FLOAT64;
        case NPY_CFLOAT: return      BH_COMPLEX64;
        case NPY_CDOUBLE: return     BH_COMPLEX128;
        default:          return     BH_UNKNOWN;
    }
};

int type_cph2py(bh_type type)
{
    switch(type)
    {
        case BH_BOOL: return    NPY_BOOL;
        case BH_INT8: return    NPY_BYTE;
        case BH_UINT8: return   NPY_UBYTE;
        case BH_INT16: return   NPY_SHORT;
        case BH_UINT16: return  NPY_USHORT;
        case BH_INT32: return   NPY_INT;
        case BH_UINT32: return  NPY_UINT;
        #if NPY_BITSOF_LONG == 32
            case BH_INT64: return   NPY_LONGLONG;
            case BH_UINT64: return  NPY_ULONGLONG;
        #else
            case BH_INT64: return  NPY_LONG;
            case BH_UINT64: return NPY_ULONG;
        #endif
        case BH_FLOAT32: return    NPY_FLOAT;
        case BH_FLOAT64: return    NPY_DOUBLE;
        case BH_COMPLEX64: return  NPY_CFLOAT;
        case BH_COMPLEX128: return NPY_CDOUBLE;
    }
	assert(32==43);
    return -1;
};


const char* bh_npy_type_text(int npy_type)
{
    switch (npy_type)
    {
    case NPY_BOOL: return "NPY_BOOL";
    case NPY_BYTE: return "NPY_BYTE";
    case NPY_UBYTE: return "NPY_UBYTE";
    case NPY_SHORT: return "NPY_SHORT";
    case NPY_USHORT: return "NPY_USHORT";
    case NPY_INT: return "NPY_INT";
    case NPY_UINT: return "NPY_UINT";
    case NPY_LONG: return "NPY_LONG";
    case NPY_ULONG: return "NPY_ULONG";
    case NPY_LONGLONG: return "NPY_LONGLONG";
    case NPY_ULONGLONG: return "NPY_ULONGLONG";
    case NPY_FLOAT: return "NPY_FLOAT";
    case NPY_DOUBLE: return "NPY_DOUBLE";
    case NPY_LONGDOUBLE: return "NPY_LONGDOUBLE";
    case NPY_CFLOAT: return "NPY_CFLOAT";
    case NPY_CDOUBLE: return "NPY_CDOUBLE";
    case NPY_CLONGDOUBLE: return "NPY_CLONGDOUBLE";
    case NPY_OBJECT: return "NPY_OBJECT";
    case NPY_STRING: return "NPY_STRING";
    case NPY_UNICODE: return "NPY_UNICODE";
    case NPY_VOID: return "NPY_VOID";
    case NPY_DATETIME: return "NPY_DATETIME";
    case NPY_TIMEDELTA: return "NPY_TIMEDELTA";
    case NPY_HALF: return "NPY_HALF";
    case NPY_NTYPES: return "NPY_NTYPES";
    case NPY_NOTYPE: return "NPY_NOTYPE";
    case NPY_CHAR: return "NPY_CHAR";
    case NPY_USERDEF: return "NPY_USERDEF";
    default: return "Unknown npy type.";

    }
}
