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
#define PPRINT_BUF_SIZE PPRINT_BUF_OPSTR_SIZE*1024

static void bh_sprint_const(const bh_instruction *instr, char buf[] ) {

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
                                 (long long) instr->constant.value.int64);
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
                        (unsigned long long) instr->constant.value.uint64);
            break;
        case BH_FLOAT32:
            sprintf(buf, "[ CONST(%s)=%f ]", bh_type_text(instr->constant.type),
                                             instr->constant.value.float32);
            break;
        case BH_FLOAT64:
            sprintf(buf, "[ CONST(%s)=%lf ]", bh_type_text(instr->constant.type),
                                              instr->constant.value.float64);
            break;
        case BH_R123:
            sprintf(buf, "[ CONST(%s)={start=%llu,key=%llu} ]",
                    bh_type_text(instr->constant.type),
                    (unsigned long long)instr->constant.value.r123.start,
                    (unsigned long long)instr->constant.value.r123.key);
            break;
        case BH_COMPLEX64:
            sprintf(buf, "[ CONST(%s)={real=%f,img=%f} ]",
                    bh_type_text(instr->constant.type),
                    instr->constant.value.complex64.real,
                    instr->constant.value.complex64.imag);
            break;
        case BH_COMPLEX128:
            sprintf(buf, "[ CONST(%s)={real=%lf,imag=%lf} ]",
                    bh_type_text(instr->constant.type),
                    instr->constant.value.complex128.real,
                    instr->constant.value.complex128.imag);
            break;
        case BH_UNKNOWN:
            sprintf(buf, "[ CONST(BH_UNKNOWN)=? ]");
        default:
            sprintf(buf, "[ CONST(?)=? ]");
    }

}



static void bh_sprint_coord( char buf[], const bh_index coord[], bh_index dims ) {

    char tmp[PPRINT_BUF_SHAPE_SIZE];
    bh_index j;

    for(j=0; j<dims; j++)
    {
        sprintf(tmp, "%lld", (long long)coord[j]);
        strcat(buf, tmp);
        if (j<dims-1) {
            strcat(buf, ", ");
        }
    }
}

/*********************************************************/
/****************** Public functions *********************/
/*********************************************************/

/* Pretty print an base.
 *
 * @op      The base in question
 * @buf     Output buffer (must have sufficient size)
 */
void bh_sprint_base(const bh_base *base, char buf[])
{
    sprintf(buf, "[ Addr: %p Type: %s #elem: %ld Data: %p ]",
            base, bh_type_text(base->type), (long) base->nelem, base->data
    );
}

/* Pretty print an view.
 *
 * @op      The view in question
 * @buf     Output buffer (must have sufficient size)
 */
void bh_sprint_view(const bh_view *op, char buf[])
{
    char    stride[PPRINT_BUF_STRIDE_SIZE]  = "?",
            shape[PPRINT_BUF_SHAPE_SIZE]    = "?",
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
        bh_sprint_base(op->base, tmp);
        sprintf(buf, "[ Dims: %d Start: %d Shape: %s Stride: %s Base=>%s]",
                (int)op->ndim, (int)op->start, shape, stride, tmp);
    }

}

/* Pretty print an instruction.
 *
 * @instr   The instruction in question
 * @buf     Output buffer (must have sufficient size)
 * @newline The new line string
 */
void bh_sprint_instr(const bh_instruction *instr, char buf[], const char newline[])
{

    char op_str[PPRINT_BUF_OPSTR_SIZE];
    char tmp[PPRINT_BUF_OPSTR_SIZE];
    int op_count = bh_operands(instr->opcode);
    int i;
    if(instr->opcode > BH_MAX_OPCODE_ID)//It is a extension method
        sprintf(buf, "Extension Method (%d) OPS=%d{%s", (int)instr->opcode, op_count, newline);
    else
        sprintf(buf, "%s OPS=%d{%s", bh_opcode_text(instr->opcode), op_count, newline);
    for(i=0; i < op_count; i++) {

        if (!bh_is_constant(&instr->operand[i]))
            bh_sprint_view(&instr->operand[i], op_str );
        else
            //sprintf(op_str, "CONSTANT");
            bh_sprint_const( instr, op_str );

        sprintf(tmp, "  OP%d %s%s", i, op_str, newline);
        strcat(buf, tmp);
    }
    strcat(buf, "}");
}

/* Pretty print an instruction.
 *
 * @instr  The instruction in question
 */
void bh_pprint_instr(const bh_instruction *instr)
{

    char buf[PPRINT_BUF_SIZE];
    bh_sprint_instr( instr, buf, "\n" );
    puts( buf );
}

/* Pretty print an instruction list.
 *
 * @instr_list  The instruction list in question
 * @ninstr      Number of instructions
 * @txt         Text prepended the instruction list,
 *              ignored when NULL
 */
void bh_pprint_instr_list(const bh_instruction instr_list[],
                          bh_intp ninstr, const char* txt)
{
    printf("%s %d {\n", txt, (int)ninstr);
    for(bh_intp i=0; i < ninstr; ++i)
        bh_pprint_instr(&instr_list[i]);
    printf("}\n");
}

/* Pretty print an array view.
 *
 * @view  The array view in question
 */
void bh_pprint_array(const bh_view *view)
{
    char buf[PPRINT_BUF_OPSTR_SIZE];
    bh_sprint_view(view, buf);
    puts(buf);
}

/* Pretty print an array base.
 *
 * @base  The array base in question
 */
void bh_pprint_base(const bh_base *base)
{
    char buf[PPRINT_BUF_OPSTR_SIZE];
    bh_sprint_base(base, buf);
    puts(buf);
}

/* Pretty print an coordinate.
 *
 * @coord  The coordinate in question
 * @ndims  Number of dimensions
 */
void bh_pprint_coord(bh_index coord[], bh_index ndims)
{
    char buf[PPRINT_BUF_SHAPE_SIZE];
    sprintf(buf, "Coord ( ");
    bh_sprint_coord(buf, coord, ndims);
    strcat(buf, " )");
    puts(buf);
}

/**
 *  Dump instruction-list to file.
 */
void bh_pprint_trace_file(const bh_ir *bhir, char trace_fn[])
{
    char instr[8000];
    FILE *file;
    file = fopen(trace_fn, "w");
    for(uint32_t i=0; i<bhir->instr_list.size(); ++i) {
        char buf[50];
        sprintf(buf, "%d: ", i);
        bh_sprint_instr(&bhir->instr_list[i], instr, "\n");
        fputs(buf, file);
        fputs(instr, file);
        fputs("\n", file);
    }
    fclose(file);
}

