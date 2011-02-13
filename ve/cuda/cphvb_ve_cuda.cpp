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

#include <cphvb.h>
#include <cuda.h>
#include "cphvb_vm_cuda.h"


cphvb_error cphvb_ve_cuda_device_count(int* count)
{
    try {
        if (deviceManager == NULL)
        {
            deviceManager = new DeviceManagerSimple();
        }
        *count = deviceManager->deviceCount();
    } 
    catch (exception& e)
    {
        return CPHVB_ERROR;
    }
    return CPHVB_SUCCESS;
}


cphvb_error cphvb_ve_cuda_init_device(int device_id)
{
    try {
        if (deviceManager == NULL)
        {
            deviceManager = new DeviceManagerSimple();
        }
        deviceManager->initDevice(device_id);
        dataManager = new DataManagerSimple();
        instructionScheduler = new instructionSchedulerSimple();
    } 
    catch (exception& e)
    {
        return CPHVB_ERROR;
    }
    return CPHVB_SUCCESS;
}

cphvb_error cphvb_ve_cuda_init()
{
    return cphvb_vm_cuda_init_device(0);
}

cphvb_error cphvb_ve_cuda_execute(cphvb_int32 instructionCount,
                                  cphvb_instruction* instruktionList)
{
    try {
        instructionScheduler->add(instruction_count,instructionList);
    }
    catch (exception& e)
    {
        return CPHVB_ERROR;
    }
    return CPHVB_SUCCESS;
}

cphvb_error cphvb_ve_cuda_shutdown()
{
    try {
        instructionScheduler->flush();
        delete instructionScheduler;
        delete dataManager;
        delete deviceManager;
    }
    catch (exception& e)
    {
        return CPHVB_ERROR;
    }
    return CPHVB_SUCCESS;
}

