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
#include "MemoryManager.hpp"

class MemoryManagerPadding : public MemoryManagerSimple
{
private:
    size_t cudaAlignment(cphvb_type type)
    {
        switch (cphvb_type_size(type))
        {
        case 1:
            return 32;
        case 2:
            return 64;
        case 4:
        case 8:
            return 128;
        default:
            return 128;
        }
    }
    
    size_t cudaSize(cphVBArray* arraySpec)
    {
        assert (arraySpec->ndim > 0);
        size_t alignment = cudaAlignment(arraySpec->type);
        int i = arraySpec->ndim - 1;

        while (arraySpec->shape[i] == 1 && i > 0)
            --i; //skip dims that are 1
        
        size_t size = arraySpec->shape[i] * cphvb_type_size(arraySpec->type);
        int padding = 0;
        if (size % alignment != 0)
        {
            padding = (alignment - (size % alignment)); 
            size += padding;
        }
        arraySpec->cudaPaddingDim = i;
        arraySpec->cudaPadding = padding / cphvb_type_size(arraySpec->type);
        arraySpec->cudaElemPerPad = arraySpec->shape[i];
        while (--i >= 0)
        {
            size *= arraySpec->shape[i];
        }
        arraySpec->cudaSize = size;
        return size;
    }


public:
    MemoryManagerSimple() {}
    
    CUdeviceptr deviceAlloc(cphVBArray* arraySpec)
    {
        assert (arraySpec->ndim > 0);
        CUdeviceprt cudaPtr;
        //size_t size = cudaSize(arraySpec);
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

    cphvb_data_ptr hostAlloc(cphVBArray arraySpec)
    {

    }

    void copyToHost(cphVBArray arraySpec)
    {

    }

    void copyToDevice(cphVBArray arraySpec)
    {

    }
    
    void setCudaStride(cphVBArray* array)
    {
        cphVBArray base = cphvb_base_array(array);
        assert(array->ndim > 0);
        int padding, i;
        bool endOnlyPadding = true;
        for (i = 0; < array->ndim ; ++i)
        {
            padding = array->stride[i] / base->cudaElemPerPad;
            array->cudaStride[i] = array->stride[i] + padding;
            if ( The padding is not only at the end of each row   )
            {
                write it in the cphVBArray struct
            }
        }
    }
}


