/*
This file is part of cphVB and copyright (c) 2012 the cphVB team:
http://cphvb.bitbucket.org

cphVB is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as 
published by the Free Software Foundation, either version 3 
of the License, or (at your option) any later version.

cphVB is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the 
GNU Lesser General Public License along with cphVB. 

If not, see <http://www.gnu.org/licenses/>.
*/
#include <stdio.h>
#include <string.h>
#include <cphvb.h>
#include <cphvb_pprint.h>

#define PPRINT_BUF_STRIDE_SIZE 50
#define PPRINT_BUF_SHAPE_SIZE 50
#define PPRINT_BUF_OPSTR_SIZE 512
#define PPRINT_BUF_SIZE PPRINT_BUF_OPSTR_SIZE*4

static void cphvb_sprint_array( cphvb_array *op, char buf[] ) {

    char    stride[PPRINT_BUF_STRIDE_SIZE]  = "?",
            shape[PPRINT_BUF_SHAPE_SIZE]    = "?",
            base[PPRINT_BUF_OPSTR_SIZE]     = "?",
            tmp[PPRINT_BUF_OPSTR_SIZE]      = "?";

    if (op == NULL) {
        sprintf(buf, "%p", op);
    } else {

        if (op->ndim > 0) {                 // Text of shape and stride
            sprintf(shape, " ");
            sprintf(stride, " ");
            for(cphvb_intp i=0; i< op->ndim; i++)
            {
                sprintf(tmp, "%d", (int)op->shape[i]);
                strcat(shape, tmp);
                if (i < op->ndim-1)
                    strcat(shape, ",");

                sprintf(tmp, "%d", (int)op->stride[i]);
                strcat(stride, tmp);
                if (i< op->ndim-1)
                    strcat(stride, ",");

            }
        }
                                            // Text of base-operand
        if (op->base != NULL) {
            cphvb_sprint_array( op->base, tmp );
            sprintf( base, "%p -->\n      %s\n", op->base, tmp  );
        } else {
            sprintf( base, "%p", op->base );
        }

        sprintf(buf, "{ Addr: %p Dims: %d Start: %d Shape: %s Stride: %s Type: %s Data: %p, Base: %s  }",
                op, (int)op->ndim, (int)op->start, shape, stride, 
                cphvb_type_text(op->type), op->data, base
        );

    }

}

static void cphvb_sprint_instr( cphvb_instruction *instr, char buf[] ) {

    char op_str[PPRINT_BUF_OPSTR_SIZE];
    char tmp[PPRINT_BUF_OPSTR_SIZE];
    int op_count = cphvb_operands(instr->opcode);
    int i;
    sprintf(buf, "%s OPS=%d{\n", cphvb_opcode_text( instr->opcode), op_count );
    for(i=0; i < op_count; i++) {

        if (!cphvb_is_constant(instr->operand[i]))
            cphvb_sprint_array( instr->operand[i], op_str );
        else 
            sprintf(op_str, "CONSTANT");

        sprintf(tmp, "  OP%d %s\n", i, op_str);
        strcat(buf, tmp);
    }

    if (instr->opcode == CPHVB_USERFUNC)
    {
        cphvb_userfunc* userfunc = instr->userfunc;
        for(i=0; i < userfunc->nout; i++) {
            cphvb_sprint_array( userfunc->operand[i], op_str );
            sprintf(tmp, "  OUT%d %s\n", i, op_str);
            strcat(buf, tmp);
            if (userfunc->operand[i]->base != NULL) {
                cphvb_sprint_array( userfunc->operand[i]->base, op_str );
                sprintf(tmp, "      %s\n", op_str);
                strcat(buf, tmp);
            }
        }
        for(i=userfunc->nout; i < userfunc->nout + userfunc->nin; i++) {
            cphvb_sprint_array( userfunc->operand[i], op_str );
            sprintf(tmp, "  IN%d %s\n", i, op_str);
            strcat(buf, tmp);
            if (userfunc->operand[i]->base != NULL) {
                cphvb_sprint_array( userfunc->operand[i]->base, op_str );
                sprintf(tmp, "      %s\n", op_str);
                strcat(buf, tmp);
            }
        }
    }
    strcat(buf, "}");

}

/* Pretty print an instruction.
 *
 * @instr  The instruction in question
 */
void cphvb_pprint_instr( cphvb_instruction *instr ) {

    char buf[PPRINT_BUF_SIZE];
    cphvb_sprint_instr( instr, buf );
    puts( buf );
}

void cphvb_pprint_instr_list( cphvb_instruction* instruction_list, cphvb_intp instruction_count, const char* txt )
{
    cphvb_intp count;
    printf("%s %d {\n", txt, (int)instruction_count);
    for(count=0; count < instruction_count; count++) {
        cphvb_pprint_instr( &instruction_list[count] );
    }
    printf("}\n");
}

void cphvb_pprint_bundle( cphvb_instruction* instruction_list, cphvb_intp instruction_count  )
{
    cphvb_pprint_instr_list( instruction_list, instruction_count, "BUNDLE");
}

/* Pretty print an array.
 *
 * @instr  The array in question
 */
void cphvb_pprint_array( cphvb_array *array ) {

    char buf[PPRINT_BUF_OPSTR_SIZE];
    cphvb_sprint_array( array, buf );
    puts( buf );
}

void cphvb_sprint_coord( char buf[], cphvb_index* coord, cphvb_index dims ) {

    char tmp[64];
    cphvb_index j;

    for(j=0; j<dims; j++)
    {
        sprintf(tmp, "%ld", (long)coord[j]);
        strcat(buf, tmp);
        if (j<dims-1) {
            strcat(buf, ", ");
        }
    }
}

void cphvb_pprint_coord( cphvb_index* coord, cphvb_index dims ) {

    char buf[1024];
    sprintf(buf, "Coord ( ");
    cphvb_sprint_coord( buf, coord, dims );
    strcat(buf, " )");
    puts(buf);

}
