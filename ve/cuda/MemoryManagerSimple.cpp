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

public:
    MemoryManagerSimple() {}
    
    CUdeviceptr deviceAlloc(cphVBArray* arraySpec)
    {
        assert (arraySpec->ndim > 0);
        CUdeviceptr cudaPtr;
        size_t size = cphvb_nelements(arraySpec->ndim, arraySpec->shape);
        size *= cphvb_type_size(arraySpec->type);
        CUresult error = cuMemAlloc(&cudaPtr, size);
        if (error !=  CUDA_SUCCESS)
        {
            std::runtime_error("Could not allocate memory on device");
        }
        arraySpec->cudaPtr = cudaPtr;
        return cudaPtr;
    }

    cphvb_data_ptr hostAlloc(cphVBArray* arraySpec)
    {
        size_t size = cphvb_nelements(arraySpec->ndim, arraySpec->shape);
        size *= cphvb_type_size(arraySpec->type);
        cphvb_data_ptr res = (cphvb_data_ptr)std::malloc(size);
        if (res == NULL)
        {
            std::runtime_error("Could not allocate memory on host");
        }
        arraySpec->data = res;
        return res;
    }

    void copyToHost(cphVBArray arraySpec)
    {

    }

    void copyToDevice(cphVBArray arraySpec)
    {

    }
    
};


