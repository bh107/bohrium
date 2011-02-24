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
 * along with cphVB.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <cassert>
#include <cphvb.h>
#include "PTXconstant.hpp"
#include "PTXconstantBuffer.hpp"

PTXconstantBuffer::PTXconstantBuffer() : 
    next(0) {} 

void PTXconstantBuffer::reset()
{
    next = 0;
}

PTXconstVal constVal(cphvb_type type,
                     cphvb_constant constant)
                     
{
    switch (type)
    {
    case CPHVB_BOOL: 
        return {(unsigned long int)constant.bool8};
    case CPHVB_INT8:
        return {(long int)constant.int8};
    case CPHVB_INT16:
        return {(long int)constant.int16};
    case CPHVB_INT32:
        return {(long int)constant.int32};
    case CPHVB_INT64:
        return {(long int)constant.int64};
    case CPHVB_UINT8:
        return {(unsigned long int)constant.uint8};
    case CPHVB_UINT16:
        return {(unsigned long int)constant.uint16};
    case CPHVB_UINT32:
        return {(unsigned long int)constant.uint32};
    case CPHVB_UINT64:
        return {(unsigned long int)constant.uint64};
    case CPHVB_FLOAT32:
        return {(unsigned long int)constant.float32};
    case CPHVB_FLOAT64:
        return {(unsigned long int)constant.float64};
    default:
        assert(false);
    }
}

PTXconstant* PTXconstantBuffer::newConstant(PTXtype type, 
                                            PTXconstVal value)
{
    constants[next].type = type;
    constants[next].value = value;
    constants[next].genTxt();
    return &constants[next++];    
}


PTXconstant* PTXconstantBuffer::newConstant(cphvb_type vbtype,
                                            cphvb_constant constant)
{
    return newConstant(PTXoperand::ptxType(vbtype), constVal(vbtype, constant));
}

