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
#include "cphvb.h"
#include "cphvb_ve_score.h"
#include "dispatch.cpp"

cphvb_error cphvb_ve_score_init(

    cphvb_com       *self

) {
    myself = self;
    return CPHVB_SUCCESS;
}

cphvb_error cphvb_ve_score_execute(

    cphvb_intp          instruction_count,
    cphvb_instruction   instruction_list[]

) {

    cphvb_error res = CPHVB_SUCCESS;

    for(cphvb_intp i=0; i<instruction_count; ++i) {

        res = dispatch( &instruction_list[i] );
        if (res != CPHVB_SUCCESS) {
            fprintf(
                stderr,
                "cphvb_ve_score_execute() encountered an error while executing: %s \
                in combination with argument types.",
                cphvb_opcode_text( instruction_list[i].opcode )
            );
            break;
        }

    }

    return res;

}

cphvb_error cphvb_ve_score_shutdown( void ) {

    return CPHVB_SUCCESS;

}

cphvb_error cphvb_ve_score_reg_func(char *lib, char *fun, cphvb_intp *id) {

    if(reduce_impl == NULL)//We only support one user-defind function
    {
        cphvb_com_get_func(myself, lib, fun, &reduce_impl);
        reduce_impl_id = *id;
    }
    return CPHVB_SUCCESS;
}

//Implementation of the user-defined funtion "reduce". Note that we
//follows the function signature defined by cphvb_userfunc_impl.
cphvb_error cphvb_reduce(cphvb_userfunc *arg)
{
    printf("cphvb_ve_score_reduce\n");
    return CPHVB_SUCCESS;
}
