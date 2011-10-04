/*
 * Copyright 2011 Troels Blum <troels@blum.dk>
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
#include <iostream>
#include <stdexcept>
#include <cphvb.h>
#include "BaseArray.hpp"

BaseArray::BaseArray(cphvb_array* spec_, ResourceManager* resourceManager_) 
    : ArrayOperand(spec_)
    , resourceManager(resourceManager_)
    , bufferType(oclType(spec_->type))
{
    assert(spec->base == NULL);
    assert(spec->ndim > 0);
    buffer = resourceManager->createBuffer(size() * oclSizeOf(bufferType));
    if (spec->data != NULL)
    {
        device = 0;
        writeEvent = resourceManager->enqueueWriteBuffer(buffer, spec->data, device);
    } 
    else 
    {
        writeEvent = resourceManager->completeEvent();
    }
}

void BaseArray::sync()
{
    if (spec->data == NULL)
    {
        spec->data = (cphvb_data_ptr)std::malloc(size() * oclSizeOf(bufferType));
        if (spec->data == NULL)
        {
            throw std::runtime_error("Could not allocate memory on host");
        }
    }
    resourceManager->readBuffer(buffer, spec->data, writeEvent, device);
}

OCLtype BaseArray::type()
{
    return bufferType;
}
