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
#include <cphvb.h>
#include "BaseArray.hpp"

BaseArray::BaseArray(cphvb_array* spec_, ResourceManager* resourceManager_) 
    : spec(spec_)
    , bufferType(oclType(spec_->type))
    , buffer(Buffer(size() * oclSizeOf(bufferType), resourceManager_))
{
    assert(spec->base == NULL);
    if (spec->data != NULL)
    {
        buffer.write(spec->data);
    } 

}

void BaseArray::sync()
{
    if (spec->data == NULL)
    {
        if (cphvb_data_malloc(spec) != CPHVB_SUCCESS)
        {
            throw std::runtime_error("Could not allocate memory on host");
        }
    }
    buffer.read(spec->data);
}

OCLtype BaseArray::type()
{
    return bufferType;
}

void BaseArray::printOn(std::ostream& os)
{
    os << "__global " << oclTypeStr(bufferType) << "*";
}

void BaseArray::addToKernel(cl::Kernel& kernel, unsigned int argIndex) const
{
    kernel.setArg(argIndex, buffer);
}
