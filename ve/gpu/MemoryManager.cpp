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


#include <iostream>
#include <stdexcept>
#include <cassert>
#include <cstdlib>
#include <cphvb.h>
#include "MemoryManager.hpp"

size_t MemoryManager::dataSize(cphVBarray* baseArray)
{
    size_t size = cphvb_nelements(baseArray->ndim, baseArray->shape);
    size *= oclSizeOf(baseArray->oclType);
    return size;
}

MemoryManager::MemoryManager(ResourceManager* resourceManager_) :
    resourceManager(resourceManager_)
{}
    
cl::Buffer MemoryManager::deviceAlloc(cphVBarray* baseArray)
{
    assert(baseArray->base == NULL);
    assert(baseArray->ndim > 0);
    size_t size = dataSize(baseArray);
    cl::Buffer buffer = resourceManager->createBuffer(size);
#ifdef DEBUG
    std::cout << "[VE GPU] createBuffer(" << size << ") -> " << (void*)buffer << std::endl;
#endif
    return buffer;
}

cphvb_data_ptr MemoryManager::hostAlloc(cphVBarray* baseArray)
{
    assert(baseArray->base == NULL);
    size_t size = dataSize(baseArray);
    cphvb_data_ptr res = (cphvb_data_ptr)std::malloc(size);
    if (res == NULL)
    {
        throw std::runtime_error("Could not allocate memory on host");
    }
    return res;
}

void MemoryManager::copyToHost(cphVBarray* baseArray)
{
    assert(baseArray->base == NULL);
    assert(baseArray->buffer != 0);
    assert(baseArray->data != NULL);
    if (oclType(baseArray->type) != baseArray->oclType)
    {
        //TODO implement type conversion
        throw std::runtime_error("copyToHost: Type conversion not implemented yet");
    } 
    size_t size = dataSize(baseArray);
#ifdef DEBUG
    std::cout << "[VE GPU] enqueueReadBuffer(" << (void*)baseArray->buffer << ", " << 
        baseArray->data << ", NULL, 0)" << std::endl;
#endif
    resourceManager->enqueueReadBuffer(baseArray->buffer, baseArray->data, NULL, 0);
}

void MemoryManager::copyToDevice(cphVBarray* baseArray)
{
    assert(baseArray->base == NULL);
    assert(baseArray->data != NULL);
    assert(baseArray->buffer != 0);
    size_t size = dataSize(baseArray);
#ifdef DEBUG
    std::cout << "[VE GPU] >enqueueWriteBuffer(" <<  (void*)baseArray->buffer << 
        ", "<< baseArray->data << "(" << baseArray << "), NULL, 0)" << std::endl;
#endif
    if (oclType(baseArray->type) != baseArray->cudaType)
    {
        //TODO implement type conversion
        throw std::runtime_error("copyToDevice: Type conversion not implemented yet");        
    }
    resourceManager->enqueueWriteBuffer(baseArray->buffer, baseArray->data, NULL, 0);
}
