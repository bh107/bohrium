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
#include "cphvb_ve_score.h"

static cphvb_com *myself = NULL;
static cphvb_userfunc_impl reduce_impl = NULL;
static cphvb_intp reduce_impl_id = 0;
static cphvb_userfunc_impl random_impl = NULL;
static cphvb_intp random_impl_id = 0;
static cphvb_userfunc_impl matmul_impl = NULL;
static cphvb_intp matmul_impl_id = 0;

cphvb_error cphvb_ve_score_init(cphvb_com *self)
{
    myself = self;
    return CPHVB_SUCCESS;
}

inline cphvb_error execute_range( cphvb_instruction* list, cphvb_intp start, cphvb_intp count) {

    cphvb_intp i, bundle_size;
    cphvb_instruction* inst[CPHVB_MAX_NO_INST];

    if (count > 1) {

        //cphvb_pprint_instr_list( &list[start], count, "POTENTIAL" );
        for(i=0; i < count; i++)
        {
            inst[i] = &list[start+i];
        }
        bundle_size = cphvb_inst_bundle( inst, count );
        //cphvb_pprint_instr_list( *inst, bundle_size, "BUNDLE" );

        for(i=0; i < bundle_size; i++)          // TODO: Make nice loop here
        {
            cphvb_compute_apply( inst[i] );
            inst[i]->status = CPHVB_INST_DONE;
        }

        for(i=bundle_size; i < count; i++)
        {
            cphvb_compute_apply( inst[i] );
            inst[i]->status = CPHVB_INST_DONE;
        }

    } else {

        for(i=start; i < (start+count); i++)
        {
            cphvb_compute_apply( &list[i] );
            list[i].status = CPHVB_INST_DONE;
        }

    }

    return CPHVB_SUCCESS;

}

cphvb_error cphvb_ve_score_execute( cphvb_intp instruction_count, cphvb_instruction* instruction_list )
{
    cphvb_intp count, nops, i, kernel_size;
    cphvb_instruction* inst;
    cphvb_error ret = CPHVB_SUCCESS;

    kernel_size = 0;
    for(count=0; count < instruction_count; count++)
    {
        inst = &instruction_list[count];

        if(inst->status == CPHVB_INST_DONE)         // SKIP instruction
        {
            execute_range( instruction_list, count-kernel_size, kernel_size );
            kernel_size = 0;

            continue;
        }

        nops = cphvb_operands(inst->opcode);        // Allocate memory for operands
        for(i=0; i<nops; i++)
        {
            if (!cphvb_is_constant(inst->operand[i]))
            {
                if (cphvb_data_malloc(inst->operand[i]) != CPHVB_SUCCESS)
                {
                    return CPHVB_OUT_OF_MEMORY;     // EXIT
                }
            }

        }

        switch(inst->opcode)                        // Dispatch instruction
        {
            case CPHVB_NONE:                        // NOOP.
            case CPHVB_DISCARD:
            case CPHVB_SYNC:
                execute_range( instruction_list, count-kernel_size, kernel_size );
                kernel_size = 0;

                inst->status = CPHVB_INST_DONE;
                break;

            case CPHVB_USERFUNC:                    // External libraries are executed

                execute_range( instruction_list, count-kernel_size, kernel_size );
                kernel_size = 0;

                if(inst->userfunc->id == reduce_impl_id)
                {
                    ret = reduce_impl(inst->userfunc, NULL);
                    inst->status = (ret == CPHVB_SUCCESS) ? CPHVB_INST_DONE : CPHVB_INST_UNDONE;
                }
                else if(inst->userfunc->id == random_impl_id)
                {
                    ret = random_impl(inst->userfunc, NULL);
                    inst->status = (ret == CPHVB_SUCCESS) ? CPHVB_INST_DONE : CPHVB_INST_UNDONE;
                }
                else                                // Unsupported userfunc
                {
                    ret = CPHVB_TYPE_NOT_SUPPORTED;
                }

                break;

            default:                                // Built-in operations
                kernel_size++;
        }
        
    }
                                                // All instructions succeeded.
    execute_range( instruction_list, count-kernel_size, kernel_size );
    kernel_size = 0;

    return ret;                                 // EXIT

}

cphvb_error cphvb_ve_score_shutdown( void )
{
    return CPHVB_SUCCESS;
}

cphvb_error cphvb_ve_score_reg_func(char *lib, char *fun, cphvb_intp *id) {

    if(strcmp("cphvb_reduce", fun) && reduce_impl == NULL)
    {
        cphvb_com_get_func(myself, lib, fun, &reduce_impl);
        reduce_impl_id = *id;
    }
    else if(strcmp("cphvb_random", fun) && random_impl == NULL)
    {
        cphvb_com_get_func(myself, lib, fun, &random_impl);
        random_impl_id = *id;
    }
    else if(strcmp("cphvb_matmul", fun) && matmul_impl == NULL)
    {
        cphvb_com_get_func(myself, lib, fun, &matmul_impl);
        matmul_impl_id = *id;
    }
    
    return CPHVB_SUCCESS;
}

cphvb_error cphvb_reduce( cphvb_userfunc *arg, void* ve_arg)
{
    return cphvb_compute_reduce( arg, ve_arg );
}

cphvb_error cphvb_random( cphvb_userfunc *arg, void* ve_arg)
{
    return cphvb_compute_random( arg, ve_arg );
}

cphvb_error cphvb_matmul( cphvb_userfunc *arg, void* ve_arg)
{
    printf("SCORE doing Matrix Multiplication!\n");
    return CPHVB_SUCCESS;
}

