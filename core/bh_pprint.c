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
#include <string.h>
#include <bh.h>
#include <bh_pprint.h>

#define PPRINT_BUF_STRIDE_SIZE 50
#define PPRINT_BUF_SHAPE_SIZE 50
#define PPRINT_BUF_OPSTR_SIZE 512
#define PPRINT_BUF_SIZE PPRINT_BUF_OPSTR_SIZE*4

static void bh_sprint_const( bh_instruction *instr, char buf[] ) {

    switch( instr->constant.type) {
        case BH_BOOL:
            sprintf(buf, "[ CONST(%s)=%uc ]", bh_type_text(instr->constant.type),
                                              instr->constant.value.bool8);
            break;
        case BH_INT8:
            sprintf(buf, "[ CONST(%s)=%d ]", bh_type_text(instr->constant.type),
                                             instr->constant.value.int8);
            break;
        case BH_INT16:
            sprintf(buf, "[ CONST(%s)=%d ]", bh_type_text(instr->constant.type),
                                             instr->constant.value.int16);
            break;
        case BH_INT32:
            sprintf(buf, "[ CONST(%s)=%d ]", bh_type_text(instr->constant.type),
                                             instr->constant.value.int32);
            break;
        case BH_INT64:
            sprintf(buf, "[ CONST(%s)=%lld ]", bh_type_text(instr->constant.type),
                                              instr->constant.value.int64);
            break;
        case BH_UINT8:
            sprintf(buf, "[ CONST(%s)=%o ]", bh_type_text(instr->constant.type),
                                             instr->constant.value.uint8);
            break;
        case BH_UINT16:
            sprintf(buf, "[ CONST(%s)=%u ]", bh_type_text(instr->constant.type),
                                             instr->constant.value.uint16);
            break;
        case BH_UINT32:
            sprintf(buf, "[ CONST(%s)=%u ]", bh_type_text(instr->constant.type),
                                             instr->constant.value.uint32);
            break;
        case BH_UINT64:
            sprintf(buf, "[ CONST(%s)=%llu ]", bh_type_text(instr->constant.type),
                                              instr->constant.value.uint64);
            break;
        case BH_FLOAT16:
            sprintf(buf, "[ CONST(%s)=%u ]", bh_type_text(instr->constant.type),
                                             instr->constant.value.float16);
            break;
        case BH_FLOAT32:
            sprintf(buf, "[ CONST(%s)=%f ]", bh_type_text(instr->constant.type),
                                             instr->constant.value.float32);
            break;
        case BH_FLOAT64:
            sprintf(buf, "[ CONST(%s)=%lf ]", bh_type_text(instr->constant.type),
                                              instr->constant.value.float64);
            break;
        case BH_COMPLEX64:
        case BH_COMPLEX128:
        case BH_UNKNOWN:

        default:
            sprintf(buf, "[ CONST=? ]");
    }

}

static void bh_sprint_array( bh_view *op, char buf[] ) {

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
            for(bh_intp i=0; i< op->ndim; i++)
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
        /*
                                            // Text of base-operand
        if (op->base != NULL) {
            bh_sprint_array( op->base, tmp );
            sprintf( base, "%p -->\n      %s\n", op->base, tmp  );
        } else {
            sprintf( base, "%p", op->base );
        }

        sprintf(buf, "[ Addr: %p Dims: %d Start: %d Shape: %s Stride: %s Type: %s Data: %p, Base: %s  ]",
                op, (int)op->ndim, (int)op->start, shape, stride,
                bh_type_text(op->type), op->data, base
        );
        */
    }

}

static void bh_sprint_instr( bh_instruction *instr, char buf[] ) {

    char op_str[PPRINT_BUF_OPSTR_SIZE];
    char tmp[PPRINT_BUF_OPSTR_SIZE];
    int op_count = bh_operands(instr->opcode);
    int i;
    sprintf(buf, "%s OPS=%d{\n", bh_opcode_text( instr->opcode), op_count );
    for(i=0; i < op_count; i++) {

        if (!bh_is_constant(&instr->operand[i]))
            bh_sprint_array( &instr->operand[i], op_str );
        else
            //sprintf(op_str, "CONSTANT");
            bh_sprint_const( instr, op_str );

        sprintf(tmp, "  OP%d %s\n", i, op_str);
        strcat(buf, tmp);
    }

    if (instr->opcode == BH_USERFUNC)
    {
        bh_userfunc* userfunc = instr->userfunc;
        for(i=0; i < userfunc->nout; i++) {
            bh_sprint_array( &userfunc->operand[i], op_str );
            sprintf(tmp, "  OUT%d %s\n", i, op_str);
            strcat(buf, tmp);
        }
        for(i=userfunc->nout; i < userfunc->nout + userfunc->nin; i++) {
            bh_sprint_array( &userfunc->operand[i], op_str );
            sprintf(tmp, "  IN%d %s\n", i, op_str);
            strcat(buf, tmp);
        }
    }
    strcat(buf, "}");

}

/* Pretty print an instruction.
 *
 * @instr  The instruction in question
 */
void bh_pprint_instr( bh_instruction *instr ) {

    char buf[PPRINT_BUF_SIZE];
    bh_sprint_instr( instr, buf );
    puts( buf );
}

void bh_pprint_instr_list( bh_instruction* instruction_list, bh_intp instruction_count, const char* txt )
{
    bh_intp count;
    printf("%s %d {\n", txt, (int)instruction_count);
    for(count=0; count < instruction_count; count++) {
        bh_pprint_instr( &instruction_list[count] );
    }
    printf("}\n");
}

void bh_pprint_bundle( bh_instruction* instruction_list, bh_intp instruction_count  )
{
    bh_pprint_instr_list( instruction_list, instruction_count, "BUNDLE");
}

/* Pretty print an array.
 *
 * @view  The array view in question
 */
void bh_pprint_array( bh_view *view ) {

    char buf[PPRINT_BUF_OPSTR_SIZE];
    bh_sprint_array( view, buf );
    puts( buf );
}

void bh_sprint_coord( char buf[], bh_index* coord, bh_index dims ) {

    char tmp[64];
    bh_index j;

    for(j=0; j<dims; j++)
    {
        sprintf(tmp, "%lld", (bh_int64)coord[j]);
        strcat(buf, tmp);
        if (j<dims-1) {
            strcat(buf, ", ");
        }
    }
}

void bh_pprint_coord( bh_index* coord, bh_index dims ) {

    char buf[1024];
    sprintf(buf, "Coord ( ");
    bh_sprint_coord( buf, coord, dims );
    strcat(buf, " )");
    puts(buf);

}
