
/**
 *  Add instruction operand as argument to kernel.
 *
 *  @param instr         The instruction whos operand should be converted.
 *  @param operand_idx   Index of the operand to represent as arg_t
 *  @param kernel        The kernel in which scope the argument will exist.
 */
int add_argument(bh_kernel_t* kernel, bh_instruction* instr, int operand_idx)
{
    int arg_idx = (kernel->nargs)++;
    if (bh_is_constant(instr->operand[operand_idx])) {
        kernel->scope[arg_idx].layout    = CONSTANT;
        kernel->scope[arg_idx].data      = &(instr->constant.value);
        kernel->scope[arg_idx].type      = bh_base_array(&instr->operand[operand_idx])->type;
        kernel->scope[arg_idx].nelem     = 1;
    } else {
        if (is_contigouos(&kernel->scope[arg_idx])) {
            kernel->scope[arg_idx].layout = CONTIGUOUS;
        } else {
            kernel->scope[arg_idx].layout = STRIDED;
        }
        kernel->scope[arg_idx].data      = bh_base_array(&instr->operand[operand_idx])->data;
        kernel->scope[arg_idx].type      = bh_base_array(&instr->operand[operand_idx])->type;
        kernel->scope[arg_idx].nelem     = bh_base_array(&instr->operand[operand_idx])->nelem;
        kernel->scope[arg_idx].ndim      = instr->operand[operand_idx].ndim;
        kernel->scope[arg_idx].start     = instr->operand[operand_idx].start;
        kernel->scope[arg_idx].shape     = instr->operand[operand_idx].shape;
        kernel->scope[arg_idx].stride    = instr->operand[operand_idx].stride;
    }
    return arg_idx;
}

/**
 *  Compose a kernel based on the instruction-nodes within a dag.
 */
static bh_error compose(bh_kernel_t* kernel, bh_ir* ir, bh_dag* dag)
{
    kernel->nargs   = 0;
    kernel->scope   = (bh_kernel_arg_t*)malloc(3*dag->nnode*sizeof(bh_kernel_arg_t));
    kernel->program = (bytecode_t*)malloc(dag->nnode*sizeof(bytecode_t));

    for (int i=0; i<dag->nnode; ++i) {
        kernel->tsig[i]  = bh_type_sig(instr);

        bh_instruction* instr = kernel->instr[i] = &ir->instr_list[dag->node_map[i]];
        int out=0, in1=0, in2=0;

        //
        // Program packing: output argument
        // NOTE: All but BH_NONE has an output which is an array
        if (instr->opcode != BH_NONE) {
            out = add_argument(kernel, instr, 0);
        }

        //
        // Program packing; operator, operand and input argument(s).
        switch (instr->opcode) {    // [OPCODE_SWITCH]

            case BH_ABSOLUTE:
                in2 = add_argument(kernel, instr, 1);

                kernel->program[i].op    = MAP;  // TAC
                kernel->program[i].oper  = ABSOLUTE;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;
            
                kernel->omask |= MAP;    // Operationmask
                break;
            case BH_ARCCOS:
                in2 = add_argument(kernel, instr, 1);

                kernel->program[i].op    = MAP;  // TAC
                kernel->program[i].oper  = ARCCOS;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;
            
                kernel->omask |= MAP;    // Operationmask
                break;
            case BH_ARCCOSH:
                in2 = add_argument(kernel, instr, 1);

                kernel->program[i].op    = MAP;  // TAC
                kernel->program[i].oper  = ARCCOSH;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;
            
                kernel->omask |= MAP;    // Operationmask
                break;
            case BH_ARCSIN:
                in2 = add_argument(kernel, instr, 1);

                kernel->program[i].op    = MAP;  // TAC
                kernel->program[i].oper  = ARCSIN;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;
            
                kernel->omask |= MAP;    // Operationmask
                break;
            case BH_ARCSINH:
                in2 = add_argument(kernel, instr, 1);

                kernel->program[i].op    = MAP;  // TAC
                kernel->program[i].oper  = ARCSINH;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;
            
                kernel->omask |= MAP;    // Operationmask
                break;
            case BH_ARCTAN:
                in2 = add_argument(kernel, instr, 1);

                kernel->program[i].op    = MAP;  // TAC
                kernel->program[i].oper  = ARCTAN;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;
            
                kernel->omask |= MAP;    // Operationmask
                break;
            case BH_ARCTANH:
                in2 = add_argument(kernel, instr, 1);

                kernel->program[i].op    = MAP;  // TAC
                kernel->program[i].oper  = ARCTANH;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;
            
                kernel->omask |= MAP;    // Operationmask
                break;
            case BH_CEIL:
                in2 = add_argument(kernel, instr, 1);

                kernel->program[i].op    = MAP;  // TAC
                kernel->program[i].oper  = CEIL;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;
            
                kernel->omask |= MAP;    // Operationmask
                break;
            case BH_COS:
                in2 = add_argument(kernel, instr, 1);

                kernel->program[i].op    = MAP;  // TAC
                kernel->program[i].oper  = COS;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;
            
                kernel->omask |= MAP;    // Operationmask
                break;
            case BH_COSH:
                in2 = add_argument(kernel, instr, 1);

                kernel->program[i].op    = MAP;  // TAC
                kernel->program[i].oper  = COSH;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;
            
                kernel->omask |= MAP;    // Operationmask
                break;
            case BH_EXP:
                in2 = add_argument(kernel, instr, 1);

                kernel->program[i].op    = MAP;  // TAC
                kernel->program[i].oper  = EXP;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;
            
                kernel->omask |= MAP;    // Operationmask
                break;
            case BH_EXP2:
                in2 = add_argument(kernel, instr, 1);

                kernel->program[i].op    = MAP;  // TAC
                kernel->program[i].oper  = EXP2;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;
            
                kernel->omask |= MAP;    // Operationmask
                break;
            case BH_EXPM1:
                in2 = add_argument(kernel, instr, 1);

                kernel->program[i].op    = MAP;  // TAC
                kernel->program[i].oper  = EXPM1;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;
            
                kernel->omask |= MAP;    // Operationmask
                break;
            case BH_FLOOR:
                in2 = add_argument(kernel, instr, 1);

                kernel->program[i].op    = MAP;  // TAC
                kernel->program[i].oper  = FLOOR;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;
            
                kernel->omask |= MAP;    // Operationmask
                break;
            case BH_IDENTITY:
                in2 = add_argument(kernel, instr, 1);

                kernel->program[i].op    = MAP;  // TAC
                kernel->program[i].oper  = IDENTITY;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;
            
                kernel->omask |= MAP;    // Operationmask
                break;
            case BH_IMAG:
                in2 = add_argument(kernel, instr, 1);

                kernel->program[i].op    = MAP;  // TAC
                kernel->program[i].oper  = IMAG;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;
            
                kernel->omask |= MAP;    // Operationmask
                break;
            case BH_INVERT:
                in2 = add_argument(kernel, instr, 1);

                kernel->program[i].op    = MAP;  // TAC
                kernel->program[i].oper  = INVERT;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;
            
                kernel->omask |= MAP;    // Operationmask
                break;
            case BH_ISINF:
                in2 = add_argument(kernel, instr, 1);

                kernel->program[i].op    = MAP;  // TAC
                kernel->program[i].oper  = ISINF;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;
            
                kernel->omask |= MAP;    // Operationmask
                break;
            case BH_ISNAN:
                in2 = add_argument(kernel, instr, 1);

                kernel->program[i].op    = MAP;  // TAC
                kernel->program[i].oper  = ISNAN;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;
            
                kernel->omask |= MAP;    // Operationmask
                break;
            case BH_LOG:
                in2 = add_argument(kernel, instr, 1);

                kernel->program[i].op    = MAP;  // TAC
                kernel->program[i].oper  = LOG;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;
            
                kernel->omask |= MAP;    // Operationmask
                break;
            case BH_LOG10:
                in2 = add_argument(kernel, instr, 1);

                kernel->program[i].op    = MAP;  // TAC
                kernel->program[i].oper  = LOG10;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;
            
                kernel->omask |= MAP;    // Operationmask
                break;
            case BH_LOG1P:
                in2 = add_argument(kernel, instr, 1);

                kernel->program[i].op    = MAP;  // TAC
                kernel->program[i].oper  = LOG1P;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;
            
                kernel->omask |= MAP;    // Operationmask
                break;
            case BH_LOG2:
                in2 = add_argument(kernel, instr, 1);

                kernel->program[i].op    = MAP;  // TAC
                kernel->program[i].oper  = LOG2;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;
            
                kernel->omask |= MAP;    // Operationmask
                break;
            case BH_LOGICAL_NOT:
                in2 = add_argument(kernel, instr, 1);

                kernel->program[i].op    = MAP;  // TAC
                kernel->program[i].oper  = LOGICAL_NOT;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;
            
                kernel->omask |= MAP;    // Operationmask
                break;
            case BH_REAL:
                in2 = add_argument(kernel, instr, 1);

                kernel->program[i].op    = MAP;  // TAC
                kernel->program[i].oper  = REAL;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;
            
                kernel->omask |= MAP;    // Operationmask
                break;
            case BH_RINT:
                in2 = add_argument(kernel, instr, 1);

                kernel->program[i].op    = MAP;  // TAC
                kernel->program[i].oper  = RINT;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;
            
                kernel->omask |= MAP;    // Operationmask
                break;
            case BH_SIN:
                in2 = add_argument(kernel, instr, 1);

                kernel->program[i].op    = MAP;  // TAC
                kernel->program[i].oper  = SIN;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;
            
                kernel->omask |= MAP;    // Operationmask
                break;
            case BH_SINH:
                in2 = add_argument(kernel, instr, 1);

                kernel->program[i].op    = MAP;  // TAC
                kernel->program[i].oper  = SINH;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;
            
                kernel->omask |= MAP;    // Operationmask
                break;
            case BH_SQRT:
                in2 = add_argument(kernel, instr, 1);

                kernel->program[i].op    = MAP;  // TAC
                kernel->program[i].oper  = SQRT;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;
            
                kernel->omask |= MAP;    // Operationmask
                break;
            case BH_TAN:
                in2 = add_argument(kernel, instr, 1);

                kernel->program[i].op    = MAP;  // TAC
                kernel->program[i].oper  = TAN;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;
            
                kernel->omask |= MAP;    // Operationmask
                break;
            case BH_TANH:
                in2 = add_argument(kernel, instr, 1);

                kernel->program[i].op    = MAP;  // TAC
                kernel->program[i].oper  = TANH;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;
            
                kernel->omask |= MAP;    // Operationmask
                break;
            case BH_TRUNC:
                in2 = add_argument(kernel, instr, 1);

                kernel->program[i].op    = MAP;  // TAC
                kernel->program[i].oper  = TRUNC;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;
            
                kernel->omask |= MAP;    // Operationmask
                break;
            case BH_ADD:
                in2 = add_argument(kernel, instr, 1);
                in2 = add_argument(kernel, instr, 2);

                kernel->program[i].op    = ZIP;  // TAC
                kernel->program[i].oper  = ADD;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;
            
                kernel->omask |= ZIP;    // Operationmask
                break;
            case BH_ARCTAN2:
                in2 = add_argument(kernel, instr, 1);
                in2 = add_argument(kernel, instr, 2);

                kernel->program[i].op    = ZIP;  // TAC
                kernel->program[i].oper  = ARCTAN2;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;
            
                kernel->omask |= ZIP;    // Operationmask
                break;
            case BH_BITWISE_AND:
                in2 = add_argument(kernel, instr, 1);
                in2 = add_argument(kernel, instr, 2);

                kernel->program[i].op    = ZIP;  // TAC
                kernel->program[i].oper  = BITWISE_AND;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;
            
                kernel->omask |= ZIP;    // Operationmask
                break;
            case BH_BITWISE_OR:
                in2 = add_argument(kernel, instr, 1);
                in2 = add_argument(kernel, instr, 2);

                kernel->program[i].op    = ZIP;  // TAC
                kernel->program[i].oper  = BITWISE_OR;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;
            
                kernel->omask |= ZIP;    // Operationmask
                break;
            case BH_BITWISE_XOR:
                in2 = add_argument(kernel, instr, 1);
                in2 = add_argument(kernel, instr, 2);

                kernel->program[i].op    = ZIP;  // TAC
                kernel->program[i].oper  = BITWISE_XOR;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;
            
                kernel->omask |= ZIP;    // Operationmask
                break;
            case BH_DIVIDE:
                in2 = add_argument(kernel, instr, 1);
                in2 = add_argument(kernel, instr, 2);

                kernel->program[i].op    = ZIP;  // TAC
                kernel->program[i].oper  = DIVIDE;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;
            
                kernel->omask |= ZIP;    // Operationmask
                break;
            case BH_EQUAL:
                in2 = add_argument(kernel, instr, 1);
                in2 = add_argument(kernel, instr, 2);

                kernel->program[i].op    = ZIP;  // TAC
                kernel->program[i].oper  = EQUAL;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;
            
                kernel->omask |= ZIP;    // Operationmask
                break;
            case BH_GREATER:
                in2 = add_argument(kernel, instr, 1);
                in2 = add_argument(kernel, instr, 2);

                kernel->program[i].op    = ZIP;  // TAC
                kernel->program[i].oper  = GREATER;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;
            
                kernel->omask |= ZIP;    // Operationmask
                break;
            case BH_GREATER_EQUAL:
                in2 = add_argument(kernel, instr, 1);
                in2 = add_argument(kernel, instr, 2);

                kernel->program[i].op    = ZIP;  // TAC
                kernel->program[i].oper  = GREATER_EQUAL;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;
            
                kernel->omask |= ZIP;    // Operationmask
                break;
            case BH_LEFT_SHIFT:
                in2 = add_argument(kernel, instr, 1);
                in2 = add_argument(kernel, instr, 2);

                kernel->program[i].op    = ZIP;  // TAC
                kernel->program[i].oper  = LEFT_SHIFT;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;
            
                kernel->omask |= ZIP;    // Operationmask
                break;
            case BH_LESS:
                in2 = add_argument(kernel, instr, 1);
                in2 = add_argument(kernel, instr, 2);

                kernel->program[i].op    = ZIP;  // TAC
                kernel->program[i].oper  = LESS;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;
            
                kernel->omask |= ZIP;    // Operationmask
                break;
            case BH_LESS_EQUAL:
                in2 = add_argument(kernel, instr, 1);
                in2 = add_argument(kernel, instr, 2);

                kernel->program[i].op    = ZIP;  // TAC
                kernel->program[i].oper  = LESS_EQUAL;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;
            
                kernel->omask |= ZIP;    // Operationmask
                break;
            case BH_LOGICAL_AND:
                in2 = add_argument(kernel, instr, 1);
                in2 = add_argument(kernel, instr, 2);

                kernel->program[i].op    = ZIP;  // TAC
                kernel->program[i].oper  = LOGICAL_AND;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;
            
                kernel->omask |= ZIP;    // Operationmask
                break;
            case BH_LOGICAL_OR:
                in2 = add_argument(kernel, instr, 1);
                in2 = add_argument(kernel, instr, 2);

                kernel->program[i].op    = ZIP;  // TAC
                kernel->program[i].oper  = LOGICAL_OR;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;
            
                kernel->omask |= ZIP;    // Operationmask
                break;
            case BH_LOGICAL_XOR:
                in2 = add_argument(kernel, instr, 1);
                in2 = add_argument(kernel, instr, 2);

                kernel->program[i].op    = ZIP;  // TAC
                kernel->program[i].oper  = LOGICAL_XOR;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;
            
                kernel->omask |= ZIP;    // Operationmask
                break;
            case BH_MAXIMUM:
                in2 = add_argument(kernel, instr, 1);
                in2 = add_argument(kernel, instr, 2);

                kernel->program[i].op    = ZIP;  // TAC
                kernel->program[i].oper  = MAXIMUM;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;
            
                kernel->omask |= ZIP;    // Operationmask
                break;
            case BH_MINIMUM:
                in2 = add_argument(kernel, instr, 1);
                in2 = add_argument(kernel, instr, 2);

                kernel->program[i].op    = ZIP;  // TAC
                kernel->program[i].oper  = MINIMUM;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;
            
                kernel->omask |= ZIP;    // Operationmask
                break;
            case BH_MOD:
                in2 = add_argument(kernel, instr, 1);
                in2 = add_argument(kernel, instr, 2);

                kernel->program[i].op    = ZIP;  // TAC
                kernel->program[i].oper  = MOD;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;
            
                kernel->omask |= ZIP;    // Operationmask
                break;
            case BH_MULTIPLY:
                in2 = add_argument(kernel, instr, 1);
                in2 = add_argument(kernel, instr, 2);

                kernel->program[i].op    = ZIP;  // TAC
                kernel->program[i].oper  = MULTIPLY;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;
            
                kernel->omask |= ZIP;    // Operationmask
                break;
            case BH_NOT_EQUAL:
                in2 = add_argument(kernel, instr, 1);
                in2 = add_argument(kernel, instr, 2);

                kernel->program[i].op    = ZIP;  // TAC
                kernel->program[i].oper  = NOT_EQUAL;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;
            
                kernel->omask |= ZIP;    // Operationmask
                break;
            case BH_POWER:
                in2 = add_argument(kernel, instr, 1);
                in2 = add_argument(kernel, instr, 2);

                kernel->program[i].op    = ZIP;  // TAC
                kernel->program[i].oper  = POWER;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;
            
                kernel->omask |= ZIP;    // Operationmask
                break;
            case BH_RIGHT_SHIFT:
                in2 = add_argument(kernel, instr, 1);
                in2 = add_argument(kernel, instr, 2);

                kernel->program[i].op    = ZIP;  // TAC
                kernel->program[i].oper  = RIGHT_SHIFT;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;
            
                kernel->omask |= ZIP;    // Operationmask
                break;
            case BH_SUBTRACT:
                in2 = add_argument(kernel, instr, 1);
                in2 = add_argument(kernel, instr, 2);

                kernel->program[i].op    = ZIP;  // TAC
                kernel->program[i].oper  = SUBTRACT;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;
            
                kernel->omask |= ZIP;    // Operationmask
                break;
            case BH_ADD_REDUCE:
                in2 = add_argument(kernel, instr, 1);
                in2 = add_argument(kernel, instr, 2);

                kernel->program[i].op    = REDUCE;  // TAC
                kernel->program[i].oper  = ADD;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;
            
                kernel->omask |= REDUCE;    // Operationmask
                break;
            case BH_BITWISE_AND_REDUCE:
                in2 = add_argument(kernel, instr, 1);
                in2 = add_argument(kernel, instr, 2);

                kernel->program[i].op    = REDUCE;  // TAC
                kernel->program[i].oper  = BITWISE_AND;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;
            
                kernel->omask |= REDUCE;    // Operationmask
                break;
            case BH_BITWISE_OR_REDUCE:
                in2 = add_argument(kernel, instr, 1);
                in2 = add_argument(kernel, instr, 2);

                kernel->program[i].op    = REDUCE;  // TAC
                kernel->program[i].oper  = BITWISE_OR;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;
            
                kernel->omask |= REDUCE;    // Operationmask
                break;
            case BH_BITWISE_XOR_REDUCE:
                in2 = add_argument(kernel, instr, 1);
                in2 = add_argument(kernel, instr, 2);

                kernel->program[i].op    = REDUCE;  // TAC
                kernel->program[i].oper  = BITWISE_XOR;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;
            
                kernel->omask |= REDUCE;    // Operationmask
                break;
            case BH_LOGICAL_AND_REDUCE:
                in2 = add_argument(kernel, instr, 1);
                in2 = add_argument(kernel, instr, 2);

                kernel->program[i].op    = REDUCE;  // TAC
                kernel->program[i].oper  = LOGICAL_AND;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;
            
                kernel->omask |= REDUCE;    // Operationmask
                break;
            case BH_LOGICAL_OR_REDUCE:
                in2 = add_argument(kernel, instr, 1);
                in2 = add_argument(kernel, instr, 2);

                kernel->program[i].op    = REDUCE;  // TAC
                kernel->program[i].oper  = LOGICAL_OR;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;
            
                kernel->omask |= REDUCE;    // Operationmask
                break;
            case BH_LOGICAL_XOR_REDUCE:
                in2 = add_argument(kernel, instr, 1);
                in2 = add_argument(kernel, instr, 2);

                kernel->program[i].op    = REDUCE;  // TAC
                kernel->program[i].oper  = LOGICAL_XOR;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;
            
                kernel->omask |= REDUCE;    // Operationmask
                break;
            case BH_MAXIMUM_REDUCE:
                in2 = add_argument(kernel, instr, 1);
                in2 = add_argument(kernel, instr, 2);

                kernel->program[i].op    = REDUCE;  // TAC
                kernel->program[i].oper  = MAXIMUM;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;
            
                kernel->omask |= REDUCE;    // Operationmask
                break;
            case BH_MINIMUM_REDUCE:
                in2 = add_argument(kernel, instr, 1);
                in2 = add_argument(kernel, instr, 2);

                kernel->program[i].op    = REDUCE;  // TAC
                kernel->program[i].oper  = MINIMUM;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;
            
                kernel->omask |= REDUCE;    // Operationmask
                break;
            case BH_MULTIPLY_REDUCE:
                in2 = add_argument(kernel, instr, 1);
                in2 = add_argument(kernel, instr, 2);

                kernel->program[i].op    = REDUCE;  // TAC
                kernel->program[i].oper  = MULTIPLY;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;
            
                kernel->omask |= REDUCE;    // Operationmask
                break;
            case BH_ADD_ACCUMULATE:
                in2 = add_argument(kernel, instr, 1);
                in2 = add_argument(kernel, instr, 2);

                kernel->program[i].op    = SCAN;  // TAC
                kernel->program[i].oper  = ADD;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;
            
                kernel->omask |= SCAN;    // Operationmask
                break;
            case BH_MULTIPLY_ACCUMULATE:
                in2 = add_argument(kernel, instr, 1);
                in2 = add_argument(kernel, instr, 2);

                kernel->program[i].op    = SCAN;  // TAC
                kernel->program[i].oper  = MULTIPLY;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;
            
                kernel->omask |= SCAN;    // Operationmask
                break;
            case BH_RANDOM:
                // This one requires special-handling... what a beaty...
                in1 = (kernel->nargs)++;                // Input
                kernel->scope[in1].layout    = CONSTANT;
                kernel->scope[in1].data      = &(instr->constant.value.r123.start);
                kernel->scope[in1].type      = BH_UINT64;
                kernel->scope[in1].nelem     = 1;

                in2 = (kernel->nargs)++;
                kernel->scope[in2].layout    = CONSTANT;
                kernel->scope[in2].data      = &(instr->constant.value.r123.key);
                kernel->scope[in2].type      = BH_UINT64;
                kernel->scope[in2].nelem     = 1;

                kernel->program[i].op    = GENERATE;  // TAC
                kernel->program[i].oper  = RANDOM;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;
            
                kernel->omask |= GENERATE;    // Operationmask
                break;
            case BH_RANGE:

                kernel->program[i].op    = GENERATE;  // TAC
                kernel->program[i].oper  = RANGE;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;
            
                kernel->omask |= GENERATE;    // Operationmask
                break;
            case BH_DISCARD:
                in2 = add_argument(kernel, instr, 1);

                kernel->program[i].op    = SYSTEM;  // TAC
                kernel->program[i].oper  = DISCARD;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;
            
                kernel->omask |= SYSTEM;    // Operationmask
                break;
            case BH_FREE:

                kernel->program[i].op    = SYSTEM;  // TAC
                kernel->program[i].oper  = FREE;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;
            
                kernel->omask |= SYSTEM;    // Operationmask
                break;
            case BH_NONE:
                in2 = add_argument(kernel, instr, 1);

                kernel->program[i].op    = SYSTEM;  // TAC
                kernel->program[i].oper  = NONE;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;
            
                kernel->omask |= SYSTEM;    // Operationmask
                break;
            case BH_SYNC:
                in2 = add_argument(kernel, instr, 1);

                kernel->program[i].op    = SYSTEM;  // TAC
                kernel->program[i].oper  = SYNC;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;
            
                kernel->omask |= SYSTEM;    // Operationmask
                break;

            default:
                if (instr->opcode>=BH_MAX_OPCODE_ID) {   // Handle extensions here

                    kernel->program[i].op   = EXTENSION; // TODO: Be clever about it
                    kernel->program[i].oper = EXT_OFFSET;
                    kernel->program[i].out  = 0;
                    kernel->program[i].in1  = 0;
                    kernel->program[i].in2  = 0;

                    cout << "Extension method." << endl;
                } else {
                    in1 = -1;
                    in2 = -2;
                    printf("compose: Err=[Unsupported instruction] {\n");
                    bh_pprint_instr(instr);
                    printf("}\n");
                    return BH_ERROR;
                }
        }
    }
    
    return BH_SUCCESS;
}
