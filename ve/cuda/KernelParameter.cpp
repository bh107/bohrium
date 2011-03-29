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
#include "KernelParameter.hpp"

KernelParameter::KernelParameter(PTXtype type_, cphvb_constant value_) :
    type(type_),
    value(value_) {}

KernelParameter::KernelParameter(PTXtype type_, CUdeviceptr value_) :
    type(type_)
{
    switch (sizeof(void*))
    {
    case 4:
        value.uint32 = value_;
        break;
    case 8:
        value.uint64 = value_;
        break;
    default:
        assert(false);
    }
}

KernelParameter::KernelParameter(PTXtype type_, int value_) :
    type(type_)
{
    value.int32 = value_;
}

KernelParameter::KernelParameter(PTXtype type_, unsigned int value_) :
    type(type_)
{
    value.uint32 = value_;
}
