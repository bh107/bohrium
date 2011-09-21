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

#include <stdexcept>
#include <cuda.h>
#include "DeviceManager.hpp"

DeviceManager::DeviceManager()
{
    CUresult error = cuInit(0);
    if (error != CUDA_SUCCESS) 
    {
        throw std::runtime_error("Could not init driver.");
    }
}

int DeviceManager::deviceCount()
{
    int res;
    CUresult error = cuDeviceGetCount(&res);
    if (error != CUDA_SUCCESS) 
    {
        throw std::runtime_error("Could not get device count.");
    }
    return res;
}
