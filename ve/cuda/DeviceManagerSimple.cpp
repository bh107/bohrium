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
#include "DeviceManagerSimple.hpp"
#include "cuda_runtime_api.h"

DeviceManagerSimple::DeviceManagerSimple() {}

void DeviceManagerSimple::initDevice(int deviceId)
{
    CUresult error = cuDeviceGet(&cuDevice, deviceId);
    if (error != CUDA_SUCCESS) 
    {
        throw std::runtime_error("Could not init device.");
    }
    error = cuCtxCreate(&cuContext, CU_CTX_SCHED_AUTO, cuDevice);
    if (error != CUDA_SUCCESS) 
    {
        throw std::runtime_error("Could not create context.");
    }
}

