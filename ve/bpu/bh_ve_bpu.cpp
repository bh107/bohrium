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
#include "bh_ve_bpu.h"

#define OUTPUT_TAG "[BPU-VE] "

// The self-reference
bh_component component;

bh_error bh_ve_bpu_init(const char *name)
{
    std::cout << "Initializing the BPU-VE" << std::endl;

    bh_error err;
    if((err = bh_component_init(&component, name)) != BH_SUCCESS)
    {
        std::cerr << OUTPUT_TAG << "Failed initializing component" << std::endl;
        return err;
    }
    
    if (component.nchildren < 1)
    {
        std::cerr << OUTPUT_TAG << "No children given, the BPU VE currently needs at least one child." << std::endl;
        return BH_ERROR;
    }

    for (int i = 0; i < component.nchildren; ++i)
    {
        bh_component_iface* child = &component.children[i];
        err = child->init(child->name);
        if (err != BH_SUCCESS)
            return err;
    }

    std::cout << OUTPUT_TAG << "Initialized the BPU-VE" << std::endl;
    return BH_SUCCESS;
}

bh_error bh_ve_bpu_execute(bh_ir* bhir)
{

    // Try to find a child that will execute this batch

    std::cout << OUTPUT_TAG << "Got a batch with " << bhir->instr_list.size() << " instructions" << std::endl;

    bh_error ret_val = BH_ERROR;
    for (int i = 0; i < component.nchildren; ++i)
    {
        bh_component_iface* child = &component.children[i];
        if (child->execute(bhir) == BH_SUCCESS)
        {
            std::cout << OUTPUT_TAG << "Child " << i << " executed the batch" << std::endl;

            return BH_SUCCESS;
        }
    }

    std::cout << OUTPUT_TAG << "No children could execute the batch, returning error" << std::endl;
    
    return ret_val;
}

bh_error bh_ve_bpu_shutdown()
{
    std::cout << OUTPUT_TAG << "Shutting down " << OUTPUT_TAG << std::endl;
    for (int i = 0; i < component.nchildren; ++i)
    {
        bh_component_iface* child = &component.children[i];
        child->shutdown();
    }
    bh_component_destroy(&component);

    std::cout << OUTPUT_TAG << "Shutdown " << OUTPUT_TAG << " complete" << std::endl;
    return BH_SUCCESS;
}

bh_error bh_ve_bpu_extmethod(const char *fun_name, 
                             bh_opcode opcode)
{
    std::cout << OUTPUT_TAG << "Extension method request for: " << fun_name << std::endl;

    bh_extmethod_impl extmethod;
    bh_error err = bh_component_extmethod(&component, fun_name, &extmethod);
    if(err != BH_SUCCESS)
        return err;

    if (extmethod != NULL)
    {
        // Execute extension method and 
    	//return BH_SUCCESS;

        return BH_EXTMETHOD_NOT_SUPPORTED;
    }
    else
	    return BH_EXTMETHOD_NOT_SUPPORTED;
}
