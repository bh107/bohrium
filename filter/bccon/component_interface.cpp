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
#define BH_TIMING_SUM
#include <bh_timing.hpp>
#include <bh_component.h>

#include "component_interface.h"
#include "reduction_chain_filter.hpp"

//
// Components
//

static bh_component myself; // Myself

// Function pointers to our child.
static bh_component_iface *child;

static bh_intp reduction_;

// The timing ID for the filter
static bh_intp exec_timing;
static bool timing;

//
// Component interface init/execute/shutdown
//

bh_error bh_filter_bccon_init(const char* name)
{
    bh_error err;
    if ((err = bh_component_init(&myself, name)) != BH_SUCCESS) {
        return err;
    }

    // For now, we have one child exactly
    if (myself.nchildren != 1) {
        fprintf(stderr, "[reduction-FILTER] Unexpected number of children, must be 1");
        return BH_ERROR;
    }

    timing = bh_component_config_lookup_bool(&myself, "timing", false);
    if (timing)
        exec_timing = bh_timer_new("[BC-Con] Execution");

    // Let us initiate the child.
    child = &myself.children[0];
    if ((err = child->init(child->name)) != 0) {
        return err;
    }

    if (BH_SUCCESS != bh_component_config_int_option(&myself, "reduction", 0, 1, &reduction_)) {
        return BH_ERROR;
    }

    return BH_SUCCESS;
}

bh_error bh_filter_bccon_shutdown(void)
{
    bh_error err = child->shutdown();
    bh_component_destroy(&myself);
    if (timing)
        bh_timer_finalize(exec_timing);
    return err;
}

bh_error bh_filter_bccon_extmethod(const char *name, bh_opcode opcode)
{
    return child->extmethod(name, opcode);
}

bh_error bh_filter_bccon_execute(bh_ir* bhir)
{
    bh_uint64 start = 0;
    if (timing)
        start = bh_timer_stamp();
    if (reduction_) {
        reduction_chain_filter(*bhir); // Run the filter
    }
    if (timing)
        bh_timer_add(exec_timing, start, bh_timer_stamp());

    return child->execute(bhir);       // Execute the filtered bhir
}
