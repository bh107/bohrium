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


#include <cuda.h>
#include <stdexcept>
#include <cassert>
#include <cstdlib>
#include <cphvb.h>
#include "MemoryManager.hpp"

class MemoryManagerSimple : public MemoryManager
{
private:
    size_t dataSize(cphVBArray* baseArray)
    {
        size_t size = cphvb_nelements(baseArray->ndim, baseArray->shape);
        size *= cphvb_type_size(baseArray->type);
        return size;
    }

public:
    MemoryManagerSimple() {}
    
    CUdeviceptr deviceAlloc(cphVBArray* baseArray)
    {
        assert(baseArray->base == NULL);
        assert(baseArray->ndim > 0);
        CUdeviceptr cudaPtr;
        size_t size = dataSize(baseArray);
        CUresult error = cuMemAlloc(&cudaPtr, size);
        if (error !=  CUDA_SUCCESS)
        {
            throw std::runtime_error("Could not allocate memory on device");
        }
        return cudaPtr;
    }

    cphvb_data_ptr hostAlloc(cphVBArray* baseArray)
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

    void copyToHost(cphVBArray* baseArray)
    {
        assert(baseArray->base == NULL);
        assert(baseArray->cudaPtr != 0);
        assert(baseArray->data != NULL);
        size_t size = dataSize(baseArray);
        CUresult error = cuMemcpyDtoH(baseArray->data, baseArray->cudaPtr, 
                                      size);
        if (error !=  CUDA_SUCCESS)
        {
            throw std::runtime_error("Could not copy to Host.");
        }
        return;
    }

    void copyToDevice(cphVBArray* baseArray)
    {
        assert(baseArray->base == NULL);
        assert(baseArray->data != NULL);
        assert(baseArray->cudaPtr != 0);
        size_t size = dataSize(baseArray);
        CUresult error = cuMemcpyHtoD(baseArray->cudaPtr, baseArray->data,
                                      size);
        if (error !=  CUDA_SUCCESS)
        {
            throw std::runtime_error("Could not copy to Device.");
        }
    }
    
    void free(cphVBArray* baseArray)
    {
        assert(baseArray->base == NULL);
        CUresult error = cuMemFree(baseArray->cudaPtr);
        if (error !=  CUDA_SUCCESS)
        {
            throw std::runtime_error("Could not free device memory.");
        }
    }
    void deviceCopy(CUdeviceptr dest,
                    cphVBArray* src)
    {
        assert(src->base == NULL);
        assert(src->cudaPtr != 0);
        size_t size = dataSize(src);
        CUresult error = cuMemcpyDtoD(dest, src->cudaPtr, size);
        if (error !=  CUDA_SUCCESS)
        {
            throw std::runtime_error("Could not copy at Device.");
        }
    }
    void memset(cphVBArray* baseArray)
    {
        assert(baseArray->base == NULL);
        assert(baseArray->cudaPtr != 0);
        size_t nelements = cphvb_nelements(baseArray->ndim, baseArray->shape);
        switch (cphvb_type_size(baseArray->type))
        {
        case 1:
            cuMemsetD8(baseArray->cudaPtr,
                       *(unsigned char*) &(baseArray->initValue),
                       nelements);            
            break;
        case 2:
            cuMemsetD16(baseArray->cudaPtr,
                        *(unsigned short*) &(baseArray->initValue),
                        nelements);
            break;
        case 4:
            cuMemsetD32(baseArray->cudaPtr,
                        *(unsigned int*) &(baseArray->initValue),
                        nelements);
            break;
        case 8:
        default:
            throw std::runtime_error(
                "Can not initialize array with data of that size.");
        }
    }    
};
