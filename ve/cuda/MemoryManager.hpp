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

#ifndef __MEMORYMANAGER_HPP
#define __MEMORYMANAGER_HPP

#include <cuda.h>
#include "cphVBarray.hpp"

class MemoryManager
{
public:
    virtual CUdeviceptr deviceAlloc(cphVBarray* baseArray) = 0;
    virtual cphvb_data_ptr hostAlloc(cphVBarray* baseArray) = 0;
    virtual void copyToHost(cphVBarray* baseArray) = 0;
    virtual void copyToDevice(cphVBarray* baseArray) = 0;
    virtual void free(cphVBarray* baseArray) = 0;
    virtual void deviceCopy(CUdeviceptr dest,
                            cphVBarray* src) = 0;
    virtual void memset(cphVBarray* baseArray) = 0;
};

#endif

