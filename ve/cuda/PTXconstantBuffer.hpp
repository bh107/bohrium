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

#ifndef __PTXCONSTANTBUFFER_HPP
#define __PTXCONSTANTBUFFER_HPP

#include "PTXconstant.hpp"

#define CONSTANTBUFFERSIZE (1024)

class PTXconstantBuffer
{
private:
    PTXconstant constants[CONSTANTBUFFERSIZE];
    int next;
public:
    PTXconstantBuffer();
    void clear();
    PTXconstant* newConstant(PTXbaseType type, 
                             PTXconstVal value);
    PTXconstant* newConstant(PTXbaseType type, 
                             long int value);
    PTXconstant* newConstant(PTXbaseType type, 
                             unsigned long int value);
    PTXconstant* newConstant(PTXbaseType type, 
                             double value);
    PTXconstant* newConstant(PTXbaseType type, 
                             CUdeviceptr value);
    PTXconstant* newConstant(cphvb_type type,
                             cphvb_constant value);
};

#endif
