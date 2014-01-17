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
#include <stdio.h>
#include <bh.h>
#include "bh_filter_range.h"

//
// Components
//
static bh_component myself; // Myself

//Function pointers to our child.
static bh_component_iface *child;

//
// Component interface init/execute/shutdown
//

bh_error bh_filter_range_init(const char* name)
{
    bh_error ret;
    ret = bh_component_init(&myself, name); // Initialize self
    if (BH_SUCCESS != ret) {
        return ret;
    }

    if (myself.nchildren != 1) {     // Check that we only have one child
        fprintf(
            stderr,
            "[RANGE-FILTER] Unexpected number of children, must be 1"
        );
        return BH_ERROR;
    }

    child = &myself.children[0];            // Initialize child
    ret = child->init(child->name);
    if(0 != ret) {
        return ret;
    }

    return BH_SUCCESS;
}

bh_error bh_filter_range_execute(bh_ir* bhir)
{
    range_filter(bhir);                     // Run the filter
    return child->execute(bhir);             // Execute the filtered bhir
}

bh_error bh_filter_range_shutdown(void)
{
    bh_error ret = child->shutdown();
    bh_component_destroy(&myself);

    return ret;
}

bh_error bh_filter_range_extmethod(const char *name, bh_opcode opcode)
{
    return child->extmethod(name, opcode);
}

