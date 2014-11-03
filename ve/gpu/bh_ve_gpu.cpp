/*
This file is part of Bohrium and copyright (c) 2012 the Bohrium
team <http://www.bh107.org>.

Bohrium is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as 
published by the Free Software Foundation, either version 3 
of the License, or (at your option) any later version.

Bohrium is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the 
GNU Lesser General Public License along with Bohrium. 

If not, see <http://www.gnu.org/licenses/>.
*/

#include <iostream>
#include <stdexcept>
#include <bh.h>
#include "bh_ve_gpu.h"
#include "InstructionScheduler.hpp"

bh_component component;
InstructionScheduler* instructionScheduler;
ResourceManager* resourceManager;

bh_error bh_ve_gpu_init(const char *name)
{
    bh_error err;
    if((err = bh_component_init(&component, name)) != BH_SUCCESS)
        return err;
    try {
        resourceManager = new ResourceManager(&component);
        instructionScheduler = new InstructionScheduler();
    } 
    catch (std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return BH_ERROR;
    }
    
    if (component.nchildren < 1)
    {
        std::cerr << "[GPU-VE] Warning: No children given. Some operations may nop be supported." << std::endl;
    }
    for (int i = 0; i < component.nchildren; ++i)
    {
        bh_component_iface* child = &component.children[i];
        err = child->init(child->name);
        if (err != BH_SUCCESS)
            return err;
    }
    return BH_SUCCESS;
}

bh_error bh_ve_gpu_execute(bh_ir* bhir)
{
    bh_error ret_val = BH_ERROR;
    try
    { 
        ret_val =  instructionScheduler->schedule(bhir);
    }
    catch (std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }
    return ret_val;
}

bh_error bh_ve_gpu_shutdown()
{
    delete instructionScheduler;
    delete resourceManager;
    for (int i = 0; i < component.nchildren; ++i)
    {
        bh_component_iface* child = &component.children[i];
        child->shutdown();
    }
    bh_component_destroy(&component);
    return BH_SUCCESS;
}

bh_error bh_ve_gpu_extmethod(const char *fun_name, 
                             bh_opcode opcode)
{
    bh_extmethod_impl extmethod;
    bh_error err = bh_component_extmethod(&component, fun_name, &extmethod);
    if(err != BH_SUCCESS)
        return err;
    if (extmethod != NULL)
    {
        instructionScheduler->registerFunction(opcode, extmethod);
    	return BH_SUCCESS;
    }
    else
	    return BH_EXTMETHOD_NOT_SUPPORTED;
}
