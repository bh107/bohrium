/*
 * Copyright 2011 Troels Blum <troels@blum.dk>
 *
 * This file is part of cphVB.
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
#include <cstdio>
#include <stdexcept>
#include "PTXconstant.hpp"

int PTXconstant::snprint(const char* prefix, 
                         char* buf, 
                         int size,
                         const char* postfix)
{
    int res;
    switch(type)
    {
    case PTX_INT:
        res = std::snprintf(buf, size, "%s%ld%s", prefix, value.i, postfix);
        break;
    case PTX_UINT:
        res = std::snprintf(buf, size, "%s%ldU%s", prefix, value.u, postfix);
        break;
    case PTX_FLOAT:
        res = std::snprintf(buf, size, "%s%#.24e%s", prefix, value.f, postfix);
        break;
    case PTX_BITS:
        res = std::snprintf(buf, size, "%s%p%s", prefix, (void*)value.a, 
                            postfix);
        break;
    default:
        assert(false);
    }
    if (res > size)
    {
        throw std::runtime_error("Not enough buffer space for printing.");
    }
    return res;
}

void PTXconstant::set(cphvb_type vbtype,
                      cphvb_constant constant)
                     
{
    type = ptxBaseType(ptxType(vbtype));
    switch (vbtype)
    {
    case CPHVB_BOOL: 
        value = {(unsigned long int)constant.bool8};
    case CPHVB_INT8:
        value = {(long int)constant.int8};
    case CPHVB_INT16:
        value = {(long int)constant.int16};
    case CPHVB_INT32:
        value = {(long int)constant.int32};
    case CPHVB_INT64:
        value = {(long int)constant.int64};
    case CPHVB_UINT8:
        value = {(unsigned long int)constant.uint8};
    case CPHVB_UINT16:
        value = {(unsigned long int)constant.uint16};
    case CPHVB_UINT32:
        value = {(unsigned long int)constant.uint32};
    case CPHVB_UINT64:
        value = {(unsigned long int)constant.uint64};
    case CPHVB_FLOAT32:
        value = {(unsigned long int)constant.float32};
    case CPHVB_FLOAT64:
        value = {(unsigned long int)constant.float64};
    default:
        assert(false);
    }
}

void PTXconstant::set(PTXbaseType type_, 
                      PTXconstVal value_)
{
    type = type;
    value = value_;
}

void PTXconstant::set(long int value_)
{
    type = PTX_INT;
    value.i = value_;
}

void PTXconstant::set(unsigned long int value_)
{
    type = PTX_UINT;
    value.u = value_;
}

void PTXconstant::set(double value_)
{
    type = PTX_FLOAT;
    value.f = value_;
}

void PTXconstant::set(CUdeviceptr value_)
{
    type = PTX_BITS;
    value.a = value_;
}
