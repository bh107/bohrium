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

#include <assert.h>
#include <stdlib.h>
#include <bh.h>
#include "OCLtype.h"
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

OCLtype oclType(bh_type vbtype)
{
    switch (vbtype)
    {
    case BH_BOOL:
        return OCL_UINT8;
    case BH_INT8:
        return OCL_INT8;
    case BH_INT16:
        return OCL_INT16;
    case BH_INT32:
        return OCL_INT32;
    case BH_INT64:
        return OCL_INT64;
    case BH_UINT8:
        return OCL_UINT8;
    case BH_UINT16:
        return OCL_UINT16;
    case BH_UINT32:
        return OCL_UINT32;
    case BH_UINT64:
        return OCL_UINT64;
    case BH_FLOAT32:
        return OCL_FLOAT32;
    case BH_FLOAT64:
        return OCL_FLOAT64;
    case BH_COMPLEX64:
        return OCL_COMPLEX64;
    case BH_COMPLEX128:
        return OCL_COMPLEX128;
    case BH_R123:
        return OCL_R123;
    default:
        assert(false);
    }
}


const char* oclTypeStr(OCLtype type)
{
    switch (type)
    {
    case OCL_INT8: return "char";
    case OCL_INT16: return "short";
    case OCL_INT32: return "int";
    case OCL_INT64: return"long";
    case OCL_UINT8: return "uchar";
    case OCL_UINT16: return "ushort";
    case OCL_UINT32: return "uint";
    case OCL_UINT64: return "ulong";
    case OCL_FLOAT32: return "float";
    case OCL_FLOAT64: return "double";
    case OCL_COMPLEX64: return "float2";
    case OCL_COMPLEX128: return "double2";
    case OCL_R123: return "ulong2";
    case OCL_UNKNOWN: return "void";
    default: assert(false);
        
    }
}

const char* oclAPItypeStr(OCLtype type)
{
    switch (type)
    {
    case OCL_INT8: return "cl_char";
    case OCL_INT16: return "cl_short";
    case OCL_INT32: return "cl_int";
    case OCL_INT64: return"cl_long";
    case OCL_UINT8: return "cl_uchar";
    case OCL_UINT16: return "cl_ushort";
    case OCL_UINT32: return "cl_uint";
    case OCL_UINT64: return "cl_ulong";
    case OCL_FLOAT32: return "cl_float";
    case OCL_FLOAT64: return "cl_double";
    case OCL_COMPLEX64: return "cl_float2";
    case OCL_COMPLEX128: return "cl_double2";
    case OCL_R123: return "cl_ulong2";
    default: assert(false);
        
    }
}

size_t oclSizeOf(OCLtype type)
{
    switch (type)
    {
    case OCL_INT8:
        return sizeof(cl_char);
    case OCL_INT16:
        return sizeof(cl_short);
    case OCL_INT32:
        return sizeof(cl_int);
    case OCL_INT64:
        return sizeof(cl_long);
    case OCL_UINT8:
        return sizeof(cl_uchar);
    case OCL_UINT16:
        return sizeof(cl_ushort);
    case OCL_UINT32:
        return sizeof(cl_uint);
    case OCL_UINT64:
        return sizeof(cl_ulong);
    case OCL_FLOAT32:
        return sizeof(cl_float);
    case OCL_FLOAT64:
        return sizeof(cl_double);
    case OCL_COMPLEX64:
        return sizeof(cl_float2);
    case OCL_COMPLEX128:
        return sizeof(cl_double2);
    case OCL_R123:
        return sizeof(cl_ulong2);
    default:
        assert(false);
    }
}

bool isFloat(OCLtype type)
{
    switch (type)
    {
    case OCL_FLOAT32:
    case OCL_FLOAT64:
        return true;
    default:
        return false;
    }
}

bool isComplex(OCLtype type)
{
    switch (type)
    {
    case OCL_COMPLEX64:
    case OCL_COMPLEX128:
        return true;
    default:
        return false;
    }
}
