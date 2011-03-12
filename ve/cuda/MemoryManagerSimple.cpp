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


#include <iostream>
#include <stdexcept>
#include <cassert>
#include <cstdlib>
#include <cuda.h>
#include <cphvb.h>
#include "MemoryManagerSimple.hpp"
#include "CUDAerrorCode.h"

size_t MemoryManagerSimple::dataSize(cphVBarray* baseArray)
{
    size_t size = cphvb_nelements(baseArray->ndim, baseArray->shape);
    size *= cphvb_type_size(baseArray->type);
    return size;
}

MemoryManagerSimple::MemoryManagerSimple() {}
    
CUdeviceptr MemoryManagerSimple::deviceAlloc(cphVBarray* baseArray)
{
    assert(baseArray->base == NULL);
    assert(baseArray->ndim > 0);
    CUdeviceptr cudaPtr;
    size_t size = dataSize(baseArray);
    CUresult error = cuMemAlloc(&cudaPtr, size);
#ifdef DEBUG
    std::cout << "[VE CUDA] deviceAlloc: " << std::endl;
    printArraySpec(baseArray);
    std::cout << "[VE CUDA] cuMemAlloc(" << (void*)cudaPtr << ", " << 
        size << ")" << std::endl;
#endif
    if (error !=  CUDA_SUCCESS)
    {
#ifdef DEBUG
        std::cout << "[VE CUDA] " << cudaErrorStr(error) << std::endl;
#endif
        throw std::runtime_error("Could not allocate memory on device");
    }
    return cudaPtr;
}

cphvb_data_ptr MemoryManagerSimple::hostAlloc(cphVBarray* baseArray)
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

void MemoryManagerSimple::copyToHost(cphVBarray* baseArray)
{
    assert(baseArray->base == NULL);
    assert(baseArray->cudaPtr != 0);
    assert(baseArray->data != NULL);
    size_t size = dataSize(baseArray);
#ifdef DEBUG
    std::cout << "[VE CUDA] copyToHost: " << std::endl;
    printArraySpec(baseArray);
    std::cout << "[VE CUDA] cuMemcpyDtoH(" << baseArray->data << ", " << 
        (void*)baseArray->cudaPtr << ", " << size << ")" << std::endl;
#endif
    CUresult error = cuMemcpyDtoH(baseArray->data, baseArray->cudaPtr, 
                                  size);
    if (error !=  CUDA_SUCCESS)
    {
#ifdef DEBUG
        std::cout << "[VE CUDA] " << cudaErrorStr(error) << std::endl;
#endif
        throw std::runtime_error("Could not copy to Host.");
    }
    return;
}

void MemoryManagerSimple::copyToDevice(cphVBarray* baseArray)
{
    assert(baseArray->base == NULL);
    assert(baseArray->data != NULL);
    assert(baseArray->cudaPtr != 0);
    size_t size = dataSize(baseArray);
#ifdef DEBUG
    std::cout << "[VE CUDA] copyToDevice: " << std::endl;
    printArraySpec(baseArray);
    std::cout << "[VE CUDA] cuMemcpyHtoD(" <<  (void*)baseArray->cudaPtr << 
        ", " << baseArray->data << ", " << size << ")" << std::endl;
#endif
    CUresult error = cuMemcpyHtoD(baseArray->cudaPtr, baseArray->data,
                                  size);
    if (error !=  CUDA_SUCCESS)
    {
#ifdef DEBUG
        std::cout << "[VE CUDA] " << cudaErrorStr(error) << std::endl;
#endif
        throw std::runtime_error("Could not copy to Device.");
    }
}

void MemoryManagerSimple::free(cphVBarray* baseArray)
{
    assert(baseArray->base == NULL);
    CUresult error = cuMemFree(baseArray->cudaPtr);
    if (error !=  CUDA_SUCCESS)
    {
#ifdef DEBUG
        std::cout << "[VE CUDA] " << cudaErrorStr(error) << std::endl;
#endif
        throw std::runtime_error("Could not free device memory.");
    }
}
    
void MemoryManagerSimple::deviceCopy(CUdeviceptr dest,
                                     cphVBarray* src)
{
    assert(src->base == NULL);
    assert(src->cudaPtr != 0);
    size_t size = dataSize(src);
    CUresult error = cuMemcpyDtoD(dest, src->cudaPtr, size);
    if (error !=  CUDA_SUCCESS)
    {
#ifdef DEBUG
        std::cout << "[VE CUDA] " << cudaErrorStr(error) << std::endl;
#endif
        throw std::runtime_error("Could not copy at Device.");
    }
}

void MemoryManagerSimple::memset(cphVBarray* baseArray)
{
    assert(baseArray->base == NULL);
    assert(baseArray->cudaPtr != 0);
    size_t nelements = cphvb_nelements(baseArray->ndim, baseArray->shape);
    switch (cphvb_type_size(baseArray->type))
    {
    case 1:
        cuMemsetD8(baseArray->cudaPtr,
                   *(unsigned char*) &(baseArray->init_value),
                   nelements);            
        break;
    case 2:
        cuMemsetD16(baseArray->cudaPtr,
                    *(unsigned short*) &(baseArray->init_value),
                    nelements);
        break;
    case 4:
        cuMemsetD32(baseArray->cudaPtr,
                    *(unsigned int*) &(baseArray->init_value),
                    nelements);
        break;
    case 8:
    default:
        throw std::runtime_error(
            "Can not initialize array with data of that size.");
    }
}    
