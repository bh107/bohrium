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

#ifndef __CPHVBARRAY_HPP
#define __CPHVBARRAY_HPP

#include <cuda.h>  
#include <cphvb_array.h>

typedef struct cphVBarray cphVBarray;
/* Matches memory layout of cphvb_array */
struct cphVBarray
{
    CPHVB_ARRAY_BASE
    //Space reserved for extra meta data.
    //Not persistent at ownership change
    CUdeviceptr      cudaPtr;
};

void printArraySpec(cphVBarray* array);

#endif
