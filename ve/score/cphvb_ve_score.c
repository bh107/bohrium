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

    cphvb_intp      *opcode_count,
    cphvb_opcode    opcode_list[CPHVB_MAX_NO_OPERANDS],
    cphvb_intp      *datatype_count,
    cphvb_type      datatype_list[CPHVB_NO_TYPES],
    cphvb_com       *self

) {

    *opcode_count = 0;
    for (cphvb_opcode oc = 0; oc <CPHVB_NO_OPCODES; ++oc) {
        opcode_list[*opcode_count] = oc;
        ++*opcode_count;
    }
    *datatype_count = 0;
    for(cphvb_type ot = 0; ot <= CPHVB_FLOAT64; ++ot) {
        datatype_list[*datatype_count] = ot;
        ++*datatype_count;
    }

    opcode_list[*opcode_count] = CPHVB_ADD | CPHVB_REDUCE;
    ++*opcode_count;
    opcode_list[*opcode_count] = CPHVB_SUBTRACT | CPHVB_REDUCE;
    ++*opcode_count;

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
