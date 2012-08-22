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

#include <cassert>
#include <stdexcept>
#include <cphvb.h>
#include "Scalar.hpp"

Scalar::Scalar(cphvb_constant constant)
    : mytype(oclType(constant.type))
{
    switch (constant.type)
    {
    case CPHVB_BOOL:
        value.uc = constant.value.bool8;
        break;
    case CPHVB_INT8:
        value.c = constant.value.int8;
        break;
    case CPHVB_INT16:
        value.s = constant.value.int16;
        break;
    case CPHVB_INT32:
        value.i = constant.value.int32;
        break;
    case CPHVB_INT64:
        value.l = constant.value.int64;
        break;
    case CPHVB_UINT8:
        value.uc = constant.value.uint8;
        break;
    case CPHVB_UINT16:
        value.us = constant.value.uint16;
        break;
    case CPHVB_UINT32:
        value.ui = constant.value.uint32;
        break;
    case CPHVB_UINT64:
        value.ul = constant.value.uint64;
        break;
    case CPHVB_FLOAT16:
        value.h = constant.value.float16;
        break;
    case CPHVB_FLOAT32:
        value.f = constant.value.float32;
        break;
    case CPHVB_FLOAT64:
        value.d = constant.value.float64;
        break;
    default:
        throw std::runtime_error("Scalar: Unknown type.");
    }
}

Scalar::Scalar(cphvb_array* spec)
    : mytype(oclType(spec->type))
{
    assert (cphvb_is_scalar(spec));
    assert (spec->data != NULL);
    switch (spec->type)
    {
    case CPHVB_BOOL:
        value.uc = *(cphvb_bool*)spec->data;
        break;
    case CPHVB_INT8:
        value.c = *(cphvb_int8*)spec->data;
        break;
    case CPHVB_INT16:
        value.s = *(cphvb_int16*)spec->data;
        break;
    case CPHVB_INT32:
        value.i = *(cphvb_int32*)spec->data;
        break;
    case CPHVB_INT64:
        value.l = *(cphvb_int64*)spec->data;
        break;
    case CPHVB_UINT8:
        value.uc = *(cphvb_uint8*)spec->data;
        break;
    case CPHVB_UINT16:
        value.us = *(cphvb_uint16*)spec->data;
        break;
    case CPHVB_UINT32:
        value.ui = *(cphvb_uint32*)spec->data;
        break;
    case CPHVB_UINT64:
        value.ul = *(cphvb_uint64*)spec->data;
        break;
    case CPHVB_FLOAT16:
        value.h = *(cphvb_float16*)spec->data;
        break;
    case CPHVB_FLOAT32:
        value.f = *(cphvb_float32*)spec->data;
        break;
    case CPHVB_FLOAT64:
        value.d = *(cphvb_float64*)spec->data;
        break;
    default:
        throw std::runtime_error("Scalar: Unknown type.");
    }
}

Scalar::Scalar(cl_long v)
    : mytype(OCL_INT64)
{
    value.l = v;
}

OCLtype Scalar::type() const
{
    return mytype;
}

void Scalar::printOn(std::ostream& os) const
{
    os << "const " << oclTypeStr(mytype);
}

void Scalar::addToKernel(cl::Kernel& kernel, unsigned int argIndex)
{
    switch(mytype)
    {
    case OCL_INT8:
        kernel.setArg(argIndex, value.c);
        break;
    case OCL_INT16:
        kernel.setArg(argIndex, value.s);
        break;
    case OCL_INT32:
        kernel.setArg(argIndex, value.i);
        break;
    case OCL_INT64:
        kernel.setArg(argIndex, value.l);
        break;
    case OCL_UINT8:
        kernel.setArg(argIndex, value.uc);
        break;
    case OCL_UINT16:
        kernel.setArg(argIndex, value.us);
        break;
    case OCL_UINT32:
        kernel.setArg(argIndex, value.ui);
        break;
    case OCL_UINT64:
        kernel.setArg(argIndex, value.ul);
        break;
    case OCL_FLOAT16:
        kernel.setArg(argIndex, value.h);
        break;
    case OCL_FLOAT32:
        kernel.setArg(argIndex, value.f);
        break;
    case OCL_FLOAT64:
        kernel.setArg(argIndex, value.d);
        break;
    default:
        assert(false);
    }    
}
