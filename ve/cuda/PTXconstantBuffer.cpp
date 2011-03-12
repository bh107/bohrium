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

#include <cphvb.h>
#include "PTXconstant.hpp"
#include "PTXconstantBuffer.hpp"

PTXconstantBuffer::PTXconstantBuffer() : 
    next(0) {} 

void PTXconstantBuffer::clear()
{
    next = 0;
}

PTXconstant* PTXconstantBuffer::newConstant(PTXbaseType type, 
                                            PTXconstVal value)
{
    constants[next].type = type;
    constants[next].value = value;
    return &constants[next++];
}

PTXconstant* PTXconstantBuffer::newConstant(PTXbaseType type, 
                                            long int value)
{
    constants[next].type = type;
    constants[next].value.i = value;
    return &constants[next++];
}

PTXconstant* PTXconstantBuffer::newConstant(PTXbaseType type, 
                                            unsigned long int value)
{
    constants[next].type = type;
    constants[next].value.u = value;
    return &constants[next++];
}

PTXconstant* PTXconstantBuffer::newConstant(PTXbaseType type, 
                                            double value)
{
    constants[next].type = type;
    constants[next].value.f = value;
    return &constants[next++];
}

PTXconstant* PTXconstantBuffer::newConstant(PTXbaseType type, 
                                            CUdeviceptr value)
{
    constants[next].type = type;
    constants[next].value.a = value;
    return &constants[next++];
}

PTXconstant* PTXconstantBuffer::newConstant(cphvb_type vbtype,
                                            cphvb_constant constant)
{
    constants[next].set(vbtype, constant);
    return &constants[next++];
}
