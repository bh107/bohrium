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
    , bufferAllocated(false)
{
    assert(spec->base == NULL);
    deviceAlloc();
    if (spec->data != NULL)
    {
        copyToDevice();
    } 
    else 
    {
        waitFor = resourceManager.completeEvent();
    }
}

void BaseArray::deviceAlloc()
{
    assert(spec->ndim > 0);
    assert(!bufferAllocated);
    buffer = resourceManager->createBuffer(size() * oclSizeOf(bufferType));
#ifdef DEBUG
    std::cout << "[VE GPU] createBuffer(" << size() << ") -> " << (void*)buffer << std::endl;
#endif
    bufferAllocated = true;
}

void BaseArray::hostAlloc()
{
    assert(spec->data == NULL);
    spec->data = (cphvb_data_ptr)std::malloc(size() * oclSizeOf(bufferType));
    if (spec->data == NULL)
    {
        throw std::runtime_error("Could not allocate memory on host");
    }
}

void BaseArray::copyToHost()
{
    assert(bufferAllocated);
    assert(spec->data != NULL);
    if (oclType(spec->type) != bufferType)
    {
        //TODO implement type conversion
        throw std::runtime_error("copyToHost: Type conversion not implemented yet");
    } 
#ifdef DEBUG
    std::cout << "[VE GPU] enqueueReadBuffer(" << (void*)buffer << ", " << 
        spec->data << ", NULL, 0)" << std::endl;
#endif
    resourceManager->enqueueReadBuffer(buffer, spec->data, NULL, 0);
}

void BaseArray::copyToDevice()
{
    assert(spec->data != NULL);
    assert(bufferAllocated);
#ifdef DEBUG
    std::cout << "[VE GPU] >enqueueWriteBuffer(" <<  (void*)buffer << 
        ", "<< spec->data << "(" << baseArray << "), NULL, 0)" << std::endl;
#endif
    if (oclType(spec->type) != bufferType)
    {
        //TODO implement type conversion
        throw std::runtime_error("copyToDevice: Type conversion not implemented yet");        
    }
    waitFor = resourceManager->enqueueWriteBuffer(buffer, spec->data, NULL, 0);
}

OCLtype BaseArray::type()
{
    return bufferType;
}
