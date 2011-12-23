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
#include "cphvb_ve_omp.h"
#include "../score/dispatch.cpp"
#include "random.c"

cphvb_error cphvb_ve_omp_init(

    cphvb_com       *self

) {
    myself = self;
    return CPHVB_SUCCESS;
}

cphvb_error cphvb_ve_omp_execute(

    cphvb_intp          instruction_count,
    cphvb_instruction   instruction_list[]

) {

    cphvb_error res = CPHVB_SUCCESS;

    for(cphvb_intp i=0; i<instruction_count; ++i) {

        res = dispatch( &instruction_list[i] );
        if (res != CPHVB_SUCCESS) {
            fprintf(
                stderr,
                "cphvb_ve_omp_execute() encountered an error while executing: %s \
                in combination with argument types.",
                cphvb_opcode_text( instruction_list[i].opcode )
            );
            break;
        }

    }

    return res;

}

cphvb_error cphvb_ve_omp_shutdown( void ) {

    return CPHVB_SUCCESS;

}

cphvb_error cphvb_ve_omp_reg_func(char *lib, char *fun, cphvb_intp *id)
{
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

    return CPHVB_SUCCESS;
}

//Implementation of the user-defined funtion "reduce". Note that we
//follows the function signature defined by cphvb_userfunc_impl.
cphvb_error cphvb_reduce(cphvb_userfunc *arg)
{
    cphvb_reduce_type *a = (cphvb_reduce_type *) arg;
    cphvb_instruction inst;
    cphvb_intp i,j;
    cphvb_index coord[CPHVB_MAXDIM];
    memset(coord, 0, a->operand[1]->ndim * sizeof(cphvb_index));

    if(cphvb_operands(a->opcode) != 3)
    {
        fprintf(stderr, "Reduce only support binary operations.\n");
        exit(-1);
    }

    //Make sure that the array memory is allocated.
    if(cphvb_malloc_array_data(a->operand[0]) != CPHVB_SUCCESS ||
       cphvb_malloc_array_data(a->operand[1]) != CPHVB_SUCCESS)
    {
        return CPHVB_OUT_OF_MEMORY;
    }

    //We need a tmp copy of the arrays.
    cphvb_array *out = a->operand[0];
    cphvb_array *in  = a->operand[1];
    cphvb_array tmp  = *in;
    cphvb_intp step = in->stride[a->axis];
    tmp.start = 0;
    j=0;
    for(i=0; i<in->ndim; ++i)//Remove the 'axis' dimension from in
        if(i != a->axis)
        {
            tmp.shape[j] = in->shape[i];
            tmp.stride[j] = in->stride[i];
            ++j;
        }
    --tmp.ndim;

    //We copy the first element to the output.
    inst.status = CPHVB_INST_UNDONE;
    inst.opcode = CPHVB_IDENTITY;
    inst.operand[0] = out;
    inst.operand[1] = &tmp;
    cphvb_error err = dispatch(&inst);
    if(err != CPHVB_SUCCESS)
        return err;
    tmp.start += step;

    //Reduce over the 'axis' dimension.
    //NB: the first element is already handled.
    inst.status = CPHVB_INST_UNDONE;
    inst.opcode = a->opcode;
    inst.operand[0] = out;
    inst.operand[1] = out;
    inst.operand[2] = &tmp;
    cphvb_intp axis_size = in->shape[a->axis];
    for(i=1; i<axis_size; ++i)
    {
        cphvb_error err = dispatch(&inst);
        if(err != CPHVB_SUCCESS)
            return err;
        tmp.start += step;
    }

    return CPHVB_SUCCESS;
}
