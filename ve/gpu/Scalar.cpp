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

#include <cassert>
#include <stdexcept>
#include <bh.h>
#include "Scalar.hpp"
#include "bh_ve_gpu.h"

Scalar::Scalar(bh_constant constant)
    : mytype(oclType(constant.type))
{
    switch (constant.type)
    {
    case BH_BOOL:
        value.uc = constant.value.bool8;
        break;
    case BH_INT8:
        value.c = constant.value.int8;
        break;
    case BH_INT16:
        value.s = constant.value.int16;
        break;
    case BH_INT32:
        value.i = constant.value.int32;
        break;
    case BH_INT64:
        value.l = constant.value.int64;
        break;
    case BH_UINT8:
        value.uc = constant.value.uint8;
        break;
    case BH_UINT16:
        value.us = constant.value.uint16;
        break;
    case BH_UINT32:
        value.ui = constant.value.uint32;
        break;
    case BH_UINT64:
        value.ul = constant.value.uint64;
        break;
    // case BH_FLOAT16:
    //     value.h = constant.value.float16;
    //     break;
    case BH_FLOAT32:
        value.f = constant.value.float32;
        break;
    case BH_FLOAT64:
        value.d = constant.value.float64;
        break;
    case BH_COMPLEX64:
        value.fcx.s0 = constant.value.complex64.real;
        value.fcx.s1 = constant.value.complex64.imag;
        break;
    case BH_COMPLEX128:
        value.dcx.s0 = constant.value.complex128.real;
        value.dcx.s1 = constant.value.complex128.imag;
        break;
    case BH_R123:
        value.r123.s0 = constant.value.r123.start;
        value.r123.s1 = constant.value.r123.key;
        break;
    default:
        throw std::runtime_error("Scalar: Unknown type.");
    }
}

Scalar::Scalar(bh_base* spec)
    : mytype(oclType(spec->type))
{
    assert (spec->nelem == 1);
    assert (spec->data != NULL);
    switch (spec->type)
    {
    case BH_BOOL:
        value.uc = *(bh_bool*)spec->data;
        break;
    case BH_INT8:
        value.c = *(bh_int8*)spec->data;
        break;
    case BH_INT16:
        value.s = *(bh_int16*)spec->data;
        break;
    case BH_INT32:
        value.i = *(bh_int32*)spec->data;
        break;
    case BH_INT64:
        value.l = *(bh_int64*)spec->data;
        break;
    case BH_UINT8:
        value.uc = *(bh_uint8*)spec->data;
        break;
    case BH_UINT16:
        value.us = *(bh_uint16*)spec->data;
        break;
    case BH_UINT32:
        value.ui = *(bh_uint32*)spec->data;
        break;
    case BH_UINT64:
        value.ul = *(bh_uint64*)spec->data;
        break;
    // case BH_FLOAT16:
    //     value.h = *(bh_float16*)spec->data;
    //     break;
    case BH_FLOAT32:
        value.f = *(bh_float32*)spec->data;
        break;
    case BH_FLOAT64:
        value.d = *(bh_float64*)spec->data;
        break;
    case BH_COMPLEX64:
        value.fcx.s0 = (*(bh_complex64*)spec->data).real;
        value.fcx.s1 = (*(bh_complex64*)spec->data).imag;
        break;
    case BH_COMPLEX128:
        value.dcx.s0 = (*(bh_complex128*)spec->data).real;
        value.dcx.s1 = (*(bh_complex128*)spec->data).imag;
        break;
    default:
        throw std::runtime_error("Scalar: Unknown type.");
    }
}

Scalar::Scalar(bh_intp ip)
{
    switch (resourceManager->intpType())
    {
    case OCL_INT32:
        mytype = OCL_INT32;
        value.i = ip;
        break;
    case OCL_INT64:
        mytype = OCL_INT64;
        value.l = ip;
        break;
    default:
        throw std::runtime_error("Scalar: Unknown intp type.");
    }    

}

OCLtype Scalar::type() const
{
    return mytype;
}

void Scalar::printOn(std::ostream& os) const
{
    os << "const " << oclTypeStr(mytype)
#ifdef DEBUG
       << "/*" << value.i << "*/"
#endif
        ;
}

void Scalar::addToKernel(cl::Kernel& kernel, unsigned int argIndex)
{
    try {
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
            // case OCL_FLOAT16:
            //     kernel.setArg(argIndex, value.h);
            //     break;
        case OCL_FLOAT32:
            kernel.setArg(argIndex, value.f);
            break;
        case OCL_FLOAT64:
            kernel.setArg(argIndex, value.d);
            break;
        case OCL_COMPLEX64:
            kernel.setArg(argIndex, value.fcx);
            break;
        case OCL_COMPLEX128:
            kernel.setArg(argIndex, value.dcx);
            break;
        case OCL_R123:
            kernel.setArg(argIndex, value.r123);
            break;
        default:
            assert(false);
        }    
    } catch (cl::Error err)
    {
        std::cerr << "ERROR: " << err.what() << "(" << err.err() << ")" << std::endl;
        throw err;
    }
}
