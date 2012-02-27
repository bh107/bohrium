/*
 * Copyright 2012 Troels Blum <troels@blum.dk>
 *
 * This file is part of cphVB <http://code.google.com/p/cphvb/>.
 *
 * cphVB is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * cphVB is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with cphVB. If not, see <http://www.gnu.org/licenses/>.
 */

#include <cassert>
#include <stdexcept>
#include "Scalar.hpp"

Scalar::Scalar(cphvb_array* sa)
{
    assert(sa->ndim == 0);
    assert(sa->base == NULL);
    assert(sa->data != NULL);
    type = oclType(sa->type);
    switch(type)
    {
    case OCL_INT8:
        value.c = *((cl_char*)sa->data);
        break;
    case OCL_INT16:
        value.s = *((cl_short*)sa->data);
        break;
    case OCL_INT32:
        value.i = *((cl_int*)sa->data);
        break;
    case OCL_INT64:
        value.l = *((cl_long*)sa->data);
        break;
    case OCL_UINT8:
        value.uc = *((cl_uchar*)sa->data);
        break;
    case OCL_UINT16:
        value.us = *((cl_ushort*)sa->data);
        break;
    case OCL_UINT32:
        value.ui = *((cl_uint*)sa->data);
        break;
    case OCL_UINT64:
        value.ul = *((cl_ulong*)sa->data);
        break;
    case OCL_FLOAT16:
        value.h = *((cl_half*)sa->data);
        break;
    case OCL_FLOAT32:
        value.f = *((cl_float*)sa->data);
        break;
    case OCL_FLOAT64:
        value.d = *((cl_double*)sa->data);
        break;
    default:
        throw std::runtime_error("Scalar: Unknown type.");

    }
}

void Scalar::addToKernel(cl::Kernel& kernel, unsigned int argIndex) const
{
    switch(type)
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
        throw std::runtime_error("Scalar: Unknown type.");
    }    
}
