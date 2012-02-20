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
#include <stdio.h>
#include <string.h>
#include <cphvb.h>
#include <cphvb_instruction.h>

void operand_to_str( cphvb_array *op, char buf[] ) {

    char    stride[1024]  = "",
            shape[1024]   = "",
            tmp[1024]    = "";

    for(int i=0; i< op->ndim; i++)
    {
        sprintf(tmp, "%d", (int)op->shape[i]);
        strcat(shape, tmp);
        if (i< op->ndim-1)
            strcat(shape, ",");

        sprintf(tmp, "%d", (int)op->stride[i]);
        strcat(stride, tmp);
        if (i< op->ndim-1)
            strcat(stride, ",");

        sprintf(buf, "%p { Base: %p Dims: %d Start: %d Shape: %s Stride: %s Data: %p }",
            op, op->base, (int)op->ndim, (int)op->start, shape, stride, op->data
        );

    }

}

void instr_to_str( cphvb_instruction *instr, char buf[] ) {

    int op_count = cphvb_operands(instr->opcode);
    char op_str[1024];
    char tmp[1024];

    sprintf(buf, "%s {\n", cphvb_opcode_text( instr->opcode) );
    for(int i=0; i < op_count; i++) {
        operand_to_str( instr->operand[i], op_str );
        sprintf(tmp, "  OP%d %s\n", i, op_str);
        strcat(buf, tmp);
    }
    strcat(buf, "}");

}

/* Pretty print an instruction.
 *
 * @instr  The instruction in question
 */
void cphvb_instr_pprint( cphvb_instruction *instr ) {

    char buf[1024];
    instr_to_str( instr, buf );
    puts( buf );

}
