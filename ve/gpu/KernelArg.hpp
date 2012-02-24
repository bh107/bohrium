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

#ifndef __KERNELARG_HPP
#define __KERNELARG_HPP

#include <CL/cl.hpp>
#include "OCLtype.h" 
#define OCL_BUFFER OCL_TYPES

class KernelArg
{
private:
    union value_t 
    {
        cl_char c;
        cl_short s;
        cl_int i;
        cl_long l;
        cl_uchar uc;
        cl_ushort us;
        cl_uint ui;
        cl_ulong ul;
        cl_half h;
        cl_float f;
        cl_double d;
        cl::Buffer* buffer;
    } value;
    OCLtype type;
public:
    KernelArg(value_t v, OCLtype t);
};

#endif
