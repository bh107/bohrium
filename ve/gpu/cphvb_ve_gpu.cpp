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
#include "cphvb_ve_gpu.h"

cphvb_error cphvb_ve_gpu_init(cphvb_component* _component)
{
    component = _component;
    try {
        resourceManager = new ResourceManager(component);
        instructionScheduler = new InstructionScheduler(resourceManager);
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
        return instructionScheduler->schedule(instruction_count, instruction_list);
    }
    catch (std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return CPHVB_ERROR;
    }
}

cphvb_error cphvb_ve_gpu_shutdown()
{
    delete instructionScheduler;
    delete resourceManager;
    return CPHVB_SUCCESS;
}

cphvb_error cphvb_ve_gpu_reg_func(char *fun, 
                                  cphvb_intp *id)
{
    cphvb_userfunc_impl userfunc;
    cphvb_component_get_func(component, fun, &userfunc);
    if (userfunc != NULL)
    {
        instructionScheduler->registerFunction(*id, userfunc);
    	return CPHVB_SUCCESS;
    }
    else
	    return CPHVB_USERFUNC_NOT_SUPPORTED;
}
