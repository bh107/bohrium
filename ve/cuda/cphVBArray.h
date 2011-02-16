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

#ifndef __CPHVBARRAY_H
#define __CPHVBARRAY_H

#include <cuda.h>  
#include <cphvb_array.h>

typedef struct cphVBArray cphVBArray;
/* Matches memory layout of cphvb_array */
struct cphVBArray
{
    //Field used by VEM to manage ownership
    cphvb_int32      owner;
    //Pointer to the base array. If NULL this is a base array
    cphVBArray*      base;
    //The type of data in the array
    cphvb_type       type;
    //Number of dimentions
    int              ndim;
    //Index of the start element (always 0 for base-array)
    cphvb_index      start;
    //Number of elements in each dimention
    cphvb_index      shape[CPHVB_MAXDIM];
    //The stride for each dimention
    cphvb_index      stride[CPHVB_MAXDIM];
    //Pointer to the actual data
    cphvb_data_ptr   data;
    //Does the array have an initial value (if not initialized)
    bool      hasInitValue;
    //The initial value
    cphvb_constant   initValue;
    //Ref Count
    int              refCount;
    //Space reserved for extra meta data.
    //Not persistent at ownership change
    CUdeviceptr      cudaPtr;
    cphvb_index      cudaStride[CPHVB_MAXDIM];
    int              cudaPaddingDim;
    int              cudaPadding;
    cphvb_index      cudaSize;
    cphvb_index      cudaElemPerPad;
};

#endif
