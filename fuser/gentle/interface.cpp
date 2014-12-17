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
#include "interface.h"
#include "fuser.h"

//
// Components
//

static bh_component myself; // Myself

//Function pointers to our child.
static bh_component_iface *child;

//
// Component interface init/execute/shutdown
//

bh_error bh_fuser_gentle_init(const char* name)
{
    bh_error err;
    if((err = bh_component_init(&myself, name)) != BH_SUCCESS)
        return err;

    //For now, we have one child exactly
    if(myself.nchildren != 1)
    {
        fprintf(stderr, "[fuser_topological-FILTER] Unexpected number of children, must be 1");
        return BH_ERROR;
    }

    //Let us initiate the child.
    child = &myself.children[0];
    if((err = child->init(child->name)) != 0)
        return err;

    return BH_SUCCESS;
}

bh_error bh_fuser_gentle_shutdown(void)
{
    bh_error err = child->shutdown();
    bh_component_destroy(&myself);
    return err;
}

bh_error bh_fuser_gentle_execute(bh_ir* bhir)
{
    fuser(*bhir);        // Run the filter
    return child->execute(bhir); // Execute the filtered bhir
}

bh_error bh_fuser_gentle_extmethod(const char *name, bh_opcode opcode)
{
    return child->extmethod(name, opcode);
}

