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
#include "bh_filter_pprint.h"

//
// Components
//

static bh_component *myself = NULL; // Myself

static bh_component **children;     // My children
static bh_init      child_init;
static bh_execute   child_execute;
static bh_shutdown  child_shutdown;
static bh_reg_func  child_reg_func;

//
// Component interface init/execute/shutdown
//

bh_error bh_filter_pprint_init(bh_component *self)
{
    bh_intp children_count;
    bh_error ret;
    myself = self;

    ret = bh_component_children(self, &children_count, &children);
    if (children_count != 1) {
        fprintf(stderr, "Unexpected number of children for filter, must be 1");
		return BH_ERROR;
    }
    if (ret != BH_SUCCESS) {
	    return ret;
    }

    child_init      = children[0]->init;    // Initialize the child
    child_execute   = children[0]->execute;
    child_shutdown  = children[0]->shutdown;
    child_reg_func  = children[0]->reg_func;

    if ((ret = child_init(children[0])) != 0) {
        return ret;
    }

    return BH_SUCCESS;
}

bh_error bh_filter_pprint_execute(bh_ir* bhir)
{
    pprint_filter(bhir);                 // Run the filter
    return child_execute(bhir);             // Execute the filtered bhir
}

bh_error bh_filter_pprint_shutdown(void)
{
    bh_error ret;

    ret = child_shutdown();                 // Shutdown child
    bh_component_free(children[0]);
    child_init     = NULL;
    child_execute  = NULL;
    child_shutdown = NULL;
    child_reg_func = NULL;
    bh_component_free_ptr(children);
    children = NULL;

    return ret;
}

bh_error bh_filter_pprint_reg_func(const char *fun, bh_intp *id)
{
    return child_reg_func(fun, id);
}

