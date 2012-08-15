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

#include <assert.h>
#include <stdlib.h>
#include <cphvb.h>
#include "OCLtype.h"
#ifdef __APPLE__
#include <Headers/cl.h>
#else
#include <CL/cl.h>
#endif

OCLtype oclType(cphvb_type vbtype)
{
    switch (vbtype)
    {
    case CPHVB_BOOL:
        return OCL_UINT8;
    case CPHVB_INT8:
        return OCL_INT8;
    case CPHVB_INT16:
        return OCL_INT16;
    case CPHVB_INT32:
        return OCL_INT32;
    case CPHVB_INT64:
        return OCL_INT64;
    case CPHVB_UINT8:
        return OCL_UINT8;
    case CPHVB_UINT16:
        return OCL_UINT16;
    case CPHVB_UINT32:
        return OCL_UINT32;
    case CPHVB_UINT64:
        return OCL_UINT64;
    case CPHVB_FLOAT16:
        return OCL_FLOAT16;
    case CPHVB_FLOAT32:
        return OCL_FLOAT32;
    case CPHVB_FLOAT64:
        return OCL_FLOAT64;
    case CPHVB_COMPLEX64:
        return OCL_COMPLEX64;
    case CPHVB_COMPLEX128:
        return OCL_COMPLEX128;
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
    case OCL_FLOAT16: return "half";
    case OCL_FLOAT32: return "float";
    case OCL_FLOAT64: return "double";
    case OCL_COMPLEX64: return "float2";
    case OCL_COMPLEX128: return "double2";
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
    case OCL_FLOAT16: return "cl_half";
    case OCL_FLOAT32: return "cl_float";
    case OCL_FLOAT64: return "cl_double";
    case OCL_COMPLEX64: return "cl_float2";
    case OCL_COMPLEX128: return "cl_double2";
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
    case OCL_FLOAT16:
        return sizeof(cl_half);
    case OCL_FLOAT32:
        return sizeof(cl_float);
    case OCL_FLOAT64:
        return sizeof(cl_double);
    case OCL_COMPLEX64:
        return sizeof(cl_float2);
    case OCL_COMPLEX128:
        return sizeof(cl_double2);
    default:
        assert(false);
    }
}

bool isFloat(OCLtype type)
{
    switch (type)
    {
    case OCL_FLOAT16:
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
