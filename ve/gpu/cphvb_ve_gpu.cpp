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

#include <iostream>
#include <stdexcept>
#include <cphvb.h>
#include <cphvb_ve_gpu.h>

InstructionScheduler* instructionScheduler;
DeviceManager* deviceManager;

cphvb_error cphvb_ve_gpu_init(cphvb_intp *opcode_count,
                               cphvb_opcode opcode_list[],
                               cphvb_intp *datatype_count,
                               cphvb_type datatype_list[])
{
    // TODO build list of supported instructions
    *opcode_count = 0;

    
    // TODO build list of supported data types
    *datatype_count = 0;

    try {
        deviceManager = createDeviceManager();
        deviceManager->initDevice(0);
        MemoryManager* memoryManager = createMemoryManager();
        DataManager* dataManager = createDataManager(memoryManager);
        KernelGenerator* kernelGenerator = createKernelGenerator();
        instructionScheduler = createInstructionScheduler(dataManager,
                                                          kernelGenerator);
    } 
    catch (std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return CPHVB_ERROR;
    }
    return CPHVB_SUCCESS;
}

cphvb_error cphvb_ve_gpu_execute(cphvb_intp instruction_count,
                                  cphvb_instruction instruction_list[])
{
    try 
    {
        instructionScheduler->schedule(instruction_count,
                                       (cphVBinstruction*)instruction_list);
    }
    catch (std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return CPHVB_ERROR;
    }
    return CPHVB_SUCCESS;
}

cphvb_error cphvb_ve_gpu_shutdown()
{
    try 
    {
        instructionScheduler->flushAll();
    }
    catch (std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return CPHVB_ERROR;
    }
    return CPHVB_SUCCESS;
}
