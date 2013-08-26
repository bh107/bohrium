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

static void bh_sprint_base(const bh_base *base, char buf[] ) {

        sprintf(buf, "[ Addr: %p Type: %s #elem: %ld Data: %p ]",
                base, bh_type_text(base->type), (long) base->nelem, base->data
        );
}

static void bh_sprint_view(const bh_view *op, char buf[] ) {

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

static void bh_sprint_instr(const bh_instruction *instr, char buf[])
{

    char op_str[PPRINT_BUF_OPSTR_SIZE];
    char tmp[PPRINT_BUF_OPSTR_SIZE];
    int op_count = bh_operands(instr->opcode);
    int i;
    sprintf(buf, "%s OPS=%d{\n", bh_opcode_text( instr->opcode), op_count );
    for(i=0; i < op_count; i++) {

        if (!bh_is_constant(&instr->operand[i]))
            bh_sprint_view( &instr->operand[i], op_str );
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
            bh_sprint_view( &userfunc->operand[i], op_str );
            sprintf(tmp, "  OUT%d %s\n", i, op_str);
            strcat(buf, tmp);
        }
        for(i=userfunc->nout; i < userfunc->nout + userfunc->nin; i++) {
            bh_sprint_view( &userfunc->operand[i], op_str );
            sprintf(tmp, "  IN%d %s\n", i, op_str);
            strcat(buf, tmp);
        }
    }
    strcat(buf, "}");

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

static void bh_sprint_dag(char buf[], const bh_ir *bhir, const bh_dag *dag)
{
    if(bhir->ninstr > 100)
    {
        sprintf(buf, "NodeMap: (%d nodes are too many to show)\n",
                (int) bhir->ninstr);
        sprintf(buf, "Adjacency Matrix: (%d rows/columns are too many to show)\n",
                (int) bhir->ninstr);
        return;
    }

    //Print the node mappings
    sprintf(buf, "NodeMap:\n");
    for(bh_intp i=0; i<dag->nnode; ++i)
        sprintf(buf+strlen(buf), "%ld => %ld\n", (long) i, (long) dag->node_map[i]);

    //Print the adjacency matrix header
    sprintf(buf+strlen(buf), "Adjacency Matrix:\n");
    sprintf(buf+strlen(buf), "  |");
    for(bh_intp i=0; i<dag->nnode; ++i)
        sprintf(buf+strlen(buf), "%2ld", (long)i);
    sprintf(buf+strlen(buf), "\n");

    //Print line between header and body
    sprintf(buf+strlen(buf), "--+");
    for(bh_intp i=0; i<dag->nnode; ++i)
        sprintf(buf+strlen(buf), "--");
    sprintf(buf+strlen(buf), "\n");

    //Print the adjacency matrix body
    for(bh_intp i=0; i<dag->nnode; ++i)
    {
        sprintf(buf+strlen(buf), "%2ld|", (long)i);
        bh_intp ncol_idx, count=0;
        const bh_intp *col_idx = bh_adjmat_get_row(&dag->adjmat, i, &ncol_idx);
        for(bh_intp j=0; j<dag->nnode; ++j)
        {
            int value = 0;
            if(ncol_idx > 0 && count < ncol_idx && j == col_idx[count])
            {
                value = 1;
                ++count;
            }
            sprintf(buf+strlen(buf), " %d",value);
        }
        sprintf(buf+strlen(buf), "\n");
    }

    //Print the adjacency matrix header
    sprintf(buf+strlen(buf), "Adjacency MatrixT:\n");
    sprintf(buf+strlen(buf), "  |");
    for(bh_intp i=0; i<dag->nnode; ++i)
        sprintf(buf+strlen(buf), "%2ld", (long)i);
    sprintf(buf+strlen(buf), "\n");

    //Print line between header and body
    sprintf(buf+strlen(buf), "--+");
    for(bh_intp i=0; i<dag->nnode; ++i)
        sprintf(buf+strlen(buf), "--");
    sprintf(buf+strlen(buf), "\n");

    //Print the adjacency matrix body
    for(bh_intp i=0; i<dag->nnode; ++i)
    {
        sprintf(buf+strlen(buf), "%2ld|", (long)i);
        bh_intp ncol_idx, count=0;
        const bh_intp *col_idx = bh_adjmat_get_col(&dag->adjmat, i, &ncol_idx);
        for(bh_intp j=0; j<dag->nnode; ++j)
        {
            int value = 0;
            if(ncol_idx > 0 && count < ncol_idx && j == col_idx[count])
            {
                value = 1;
                ++count;
            }
            sprintf(buf+strlen(buf), " %d",value);
        }
        sprintf(buf+strlen(buf), "\n");
    }
}

static void bh_sprint_bhir(char buf[], const bh_ir *bhir)
{
    if(bhir->ninstr > 100)
    {
        sprintf(buf, "Instruction list (%d): {...} (too large to show)\n",
                (int) bhir->ninstr);
        sprintf(buf+strlen(buf), "DAG list (%d): {...} (too large to show)\n",
                (int) bhir->ndag);
        return;
    }

    sprintf(buf, "Instruction list (%d): {\n", (int) bhir->ninstr);
    for(bh_intp i=0; i < bhir->ninstr; ++i)
    {
        sprintf(buf+strlen(buf), "%3ld: ", (long) i);
        bh_sprint_instr(&bhir->instr_list[i], buf+strlen(buf));
    }
    sprintf(buf+strlen(buf), "}\n");

    sprintf(buf+strlen(buf), "DAG list (%d): {\n", (int) bhir->ndag);
    for(bh_intp i=0; i < bhir->ndag; ++i)
    {
        sprintf(buf+strlen(buf), "*****%3ld  *****\n", (long) i);
        bh_sprint_dag(buf+strlen(buf), bhir, &bhir->dag_list[i]);
    }
    sprintf(buf+strlen(buf), "}\n");
}

/*********************************************************/
/****************** Public functions *********************/
/*********************************************************/


/* Pretty print an instruction.
 *
 * @instr  The instruction in question
 */
void bh_pprint_instr(const bh_instruction *instr)
{

    char buf[PPRINT_BUF_SIZE];
    bh_sprint_instr( instr, buf );
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


/* Pretty print an BhIR DAG.
 *
 * @bhir The BhIR in question
 * @dag  The DAG in question
 *
 */
void bh_pprint_dag(const bh_ir *bhir, const bh_dag *dag)
{
    char buf[PPRINT_BUF_SIZE];
    bh_sprint_dag(buf, bhir, dag);
    puts(buf);
}

/* Pretty print an BhIR.
 *
 * @bhir The BhIR in question
 *
 */
void bh_pprint_bhir(const bh_ir *bhir)
{
    char buf[PPRINT_BUF_SIZE];
    bh_sprint_bhir(buf, bhir);
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
    for(bh_intp i=0; i<bhir->ninstr; ++i) { 
        bh_sprint_instr(&bhir->instr_list[i], instr);
        fputs(instr, file);
        fputs("\n", file);
    }
    fclose(file);
}
