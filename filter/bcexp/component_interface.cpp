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
#include "expander.hpp"

//
// Components
//

static bh_component myself;         // Myself
static bh_component_iface *child;   // My child

//The timing ID for the filter
static bh_intp exec_timing;
static bool timing;

static bohrium::filter::composite::Expander* expander = NULL;

//
// Component interface init/execute/shutdown
//

bh_error bh_filter_bcexp_init(const char* name)
{
    bh_error err;                   // Initialize myself
    if ((err = bh_component_init(&myself, name)) != BH_SUCCESS) {
        return err;
    }

    if (myself.nchildren != 1) {    // For now only one child is supported
        fprintf(stderr,
                "[FILTER-bcexp] Only a single child is supported, has %d.",
                (int)myself.nchildren);
        return BH_ERROR;
    }

    timing = bh_component_config_lookup_bool(&myself, "timing", false);
    if (timing)
        exec_timing = bh_timer_new("[BC-Exp] Execution");

    child = &myself.children[0];    // Initiate child
    if ((err = child->init(child->name)) != 0) {
        return err;
    }

    bh_intp gc_threshold, sign, matmul, powk,reduce1d;
    if ((BH_SUCCESS!=bh_component_config_int_option(&myself, "gc_threshold", 0, 2000, &gc_threshold)) or \
        (BH_SUCCESS!=bh_component_config_int_option(&myself, "matmul", 0, 1, &matmul)) or \
        (BH_SUCCESS!=bh_component_config_int_option(&myself, "sign", 0, 1, &sign)) or \
        (BH_SUCCESS!=bh_component_config_int_option(&myself, "powk", 0, 1, &powk))) {
        return BH_ERROR;
    }
    reduce1d = bh_component_config_lookup_int(&myself, "reduce1d", 0);
    try {                           // Construct the expander
        expander = new bohrium::filter::composite::Expander(gc_threshold,
                                                            matmul,
                                                            sign,
                                                            powk,
                                                            reduce1d);
    } catch (std::bad_alloc& ba) {
        fprintf(stderr, "Failed constructing Expander due to allocation error.\n");
    }
    if (NULL == expander) {
        return BH_ERROR;
    } else {
        return BH_SUCCESS;
    }
}

bh_error bh_filter_bcexp_shutdown(void)
{
    bh_error err = child->shutdown();
    bh_component_destroy(&myself);

    delete expander;
    expander = NULL;

    if (timing)
        bh_timer_finalize(exec_timing);

    return err;
}

bh_error bh_filter_bcexp_extmethod(const char *name, bh_opcode opcode)
{
    return child->extmethod(name, opcode);
}

bh_error bh_filter_bcexp_execute(bh_ir* bhir)
{
    bh_uint64 start = 0;
    if (timing)
        start = bh_timer_stamp();
    expander->expand(*bhir);                // Expand composites
    if (timing)
        bh_timer_add(exec_timing, start, bh_timer_stamp());
    bh_error res = child->execute(bhir);    // Send the bhir down the stack
    if (timing)
        start = bh_timer_stamp();
    expander->gc();                         // Collect garbage
    if (timing)
        bh_timer_add(exec_timing, start, bh_timer_stamp());
    return res;                             // Send result up the stack
}
