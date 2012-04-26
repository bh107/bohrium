/*
 * Copyright 2011 Simon A. F. Lund <safl@safl.dk>
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
 * along with cphVB. If not, see <http://www.gnu.org/licenses/>.
 */
#include <cphvb.h>
#include "cphvb_ve_simple.h"
#include <assert.h>
#include <cphvb_compute.h>
#include <cphvb_compute_reduce.h>
#include <cphvb_compute_random.h>

static cphvb_com *myself = NULL;
static cphvb_userfunc_impl reduce_impl = NULL;
static cphvb_intp reduce_impl_id = 0;
static cphvb_userfunc_impl random_impl = NULL;
static cphvb_intp random_impl_id = 0;

cphvb_error cphvb_ve_simple_init(cphvb_com *self)
{
    myself = self;

    return CPHVB_SUCCESS;
}

cphvb_error cphvb_ve_simple_execute(

    cphvb_intp          instruction_count,
    cphvb_instruction*   instruction_list

) {
    cphvb_intp count=0, nops, i;

    while(count < instruction_count)
    {
        cphvb_instruction* inst = &instruction_list[count];

        // Make sure that the array memory is allocated.
        nops = cphvb_operands(inst->opcode);
        for(i = 0; i<nops; i++) {

            if (!cphvb_is_constant(inst->operand[i])) {
                if (cphvb_data_malloc(inst->operand[i]) != CPHVB_SUCCESS) {
                    return CPHVB_OUT_OF_MEMORY;
                }
            }

        }

        // Execute it...
        switch(inst->opcode)
        {
            case CPHVB_NONE:
            case CPHVB_DISCARD:
            case CPHVB_SYNC:
                break;
            case CPHVB_USERFUNC:

                if(inst->userfunc->id == reduce_impl_id)
                {
                    inst->status = reduce_impl(inst->userfunc, NULL);
                }
                else if(inst->userfunc->id == random_impl_id)
                {
                    inst->status = random_impl(inst->userfunc, NULL);
                }
                else// Unsupported instruction
                {
                    inst->status = CPHVB_TYPE_NOT_SUPPORTED;
                }

            default://This is a regular operation.
                if(inst->status == CPHVB_INST_UNDONE) {
                    inst->status = cphvb_compute_apply( inst );
                }
        }
        ++count;

    }
    // TODO: add partial success / error / etc
    return CPHVB_SUCCESS;

}

cphvb_error cphvb_ve_simple_shutdown( void )
{
    return CPHVB_SUCCESS;
}

cphvb_error cphvb_ve_score_reg_func(char *lib, char *fun, cphvb_intp *id) {

    if(strcmp("cphvb_reduce", fun) && reduce_impl == NULL)
    {
        cphvb_com_get_func(myself, lib, "cphvb_compute_reduce", &reduce_impl);
        reduce_impl_id = *id;
    }
    else if(strcmp("cphvb_random", fun) && random_impl == NULL)
    {
        cphvb_com_get_func(myself, lib, "cphvb_compute_random", &random_impl);
        random_impl_id = *id;
    }
    return CPHVB_SUCCESS;
}

