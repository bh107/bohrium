
typedef struct block {
    int ninstr;                 // Number of instructions in block

    int tsig[10];               // Typesignature of the instructions
    int lmask[10];              // Layoutmask of the instructions

    bh_instruction* instr[10];  // Pointers to instructions
    tac_t* program;             // Ordered list of TACs
    int nargs;                  // Number of arguments to the block
    block_arg_t* scope;         // Array of block arguments

    uint32_t omask;             // Mask of the OPERATIONS in the block
    string symbol;              // Textual representation of the block
} block_t;                      // Meta-data to construct and execute a block-function

/**
 *  Add instruction operand as argument to block.
 *
 *  @param instr         The instruction whos operand should be converted.
 *  @param operand_idx   Index of the operand to represent as arg_t
 *  @param block        The block in which scope the argument will exist.
 */
static int add_argument(block_t* block, bh_instruction* instr, int operand_idx)
{
    int arg_idx = (block->nargs)++;
    if (bh_is_constant(&instr->operand[operand_idx])) {
        block->scope[arg_idx].layout    = CONSTANT;
        block->scope[arg_idx].data      = &(instr->constant.value);
        block->scope[arg_idx].type      = bh_base_array(&instr->operand[operand_idx])->type;
        block->scope[arg_idx].nelem     = 1;
    } else {
        if (is_contiguous(&block->scope[arg_idx])) {
            block->scope[arg_idx].layout = CONTIGUOUS;
        } else {
            block->scope[arg_idx].layout = STRIDED;
        }
        block->scope[arg_idx].data      = bh_base_array(&instr->operand[operand_idx])->data;
        block->scope[arg_idx].type      = bh_base_array(&instr->operand[operand_idx])->type;
        block->scope[arg_idx].nelem     = bh_base_array(&instr->operand[operand_idx])->nelem;
        block->scope[arg_idx].ndim      = instr->operand[operand_idx].ndim;
        block->scope[arg_idx].start     = instr->operand[operand_idx].start;
        block->scope[arg_idx].shape     = instr->operand[operand_idx].shape;
        block->scope[arg_idx].stride    = instr->operand[operand_idx].stride;
    }
    return arg_idx;
}

/**
 *  Compose a block based on the instruction-nodes within a dag.
 */
static bh_error compose(block_t* block, bh_ir* ir, bh_dag* dag)
{
    block->nargs   = 0;
    block->scope   = (block_arg_t*)malloc(3*dag->nnode*sizeof(block_arg_t));
    block->program = (tac_t*)malloc(dag->nnode*sizeof(tac_t));

    for (int i=0; i<dag->nnode; ++i) {
        bh_instruction* instr = block->instr[i] = &ir->instr_list[dag->node_map[i]];
        block->tsig[i] = bh_type_sig(instr);
        int out=0, in1=0, in2=0;

        //
        // Program packing: output argument
        // NOTE: All but BH_NONE has an output which is an array
        if (instr->opcode != BH_NONE) {
            out = add_argument(block, instr, 0);
        }

        //
        // Program packing; operator, operand and input argument(s).
        switch (instr->opcode) {    // [OPCODE_SWITCH]

            case BH_ABSOLUTE:
                in2 = add_argument(block, instr, 1);

                block->program[i].op    = MAP;  // TAC
                block->program[i].oper  = ABSOLUTE;
                block->program[i].out   = out;
                block->program[i].in1   = in1;
                block->program[i].in2   = in2;
            
                block->omask |= MAP;    // Operationmask
                break;
            case BH_ARCCOS:
                in2 = add_argument(block, instr, 1);

                block->program[i].op    = MAP;  // TAC
                block->program[i].oper  = ARCCOS;
                block->program[i].out   = out;
                block->program[i].in1   = in1;
                block->program[i].in2   = in2;
            
                block->omask |= MAP;    // Operationmask
                break;
            case BH_ARCCOSH:
                in2 = add_argument(block, instr, 1);

                block->program[i].op    = MAP;  // TAC
                block->program[i].oper  = ARCCOSH;
                block->program[i].out   = out;
                block->program[i].in1   = in1;
                block->program[i].in2   = in2;
            
                block->omask |= MAP;    // Operationmask
                break;
            case BH_ARCSIN:
                in2 = add_argument(block, instr, 1);

                block->program[i].op    = MAP;  // TAC
                block->program[i].oper  = ARCSIN;
                block->program[i].out   = out;
                block->program[i].in1   = in1;
                block->program[i].in2   = in2;
            
                block->omask |= MAP;    // Operationmask
                break;
            case BH_ARCSINH:
                in2 = add_argument(block, instr, 1);

                block->program[i].op    = MAP;  // TAC
                block->program[i].oper  = ARCSINH;
                block->program[i].out   = out;
                block->program[i].in1   = in1;
                block->program[i].in2   = in2;
            
                block->omask |= MAP;    // Operationmask
                break;
            case BH_ARCTAN:
                in2 = add_argument(block, instr, 1);

                block->program[i].op    = MAP;  // TAC
                block->program[i].oper  = ARCTAN;
                block->program[i].out   = out;
                block->program[i].in1   = in1;
                block->program[i].in2   = in2;
            
                block->omask |= MAP;    // Operationmask
                break;
            case BH_ARCTANH:
                in2 = add_argument(block, instr, 1);

                block->program[i].op    = MAP;  // TAC
                block->program[i].oper  = ARCTANH;
                block->program[i].out   = out;
                block->program[i].in1   = in1;
                block->program[i].in2   = in2;
            
                block->omask |= MAP;    // Operationmask
                break;
            case BH_CEIL:
                in2 = add_argument(block, instr, 1);

                block->program[i].op    = MAP;  // TAC
                block->program[i].oper  = CEIL;
                block->program[i].out   = out;
                block->program[i].in1   = in1;
                block->program[i].in2   = in2;
            
                block->omask |= MAP;    // Operationmask
                break;
            case BH_COS:
                in2 = add_argument(block, instr, 1);

                block->program[i].op    = MAP;  // TAC
                block->program[i].oper  = COS;
                block->program[i].out   = out;
                block->program[i].in1   = in1;
                block->program[i].in2   = in2;
            
                block->omask |= MAP;    // Operationmask
                break;
            case BH_COSH:
                in2 = add_argument(block, instr, 1);

                block->program[i].op    = MAP;  // TAC
                block->program[i].oper  = COSH;
                block->program[i].out   = out;
                block->program[i].in1   = in1;
                block->program[i].in2   = in2;
            
                block->omask |= MAP;    // Operationmask
                break;
            case BH_EXP:
                in2 = add_argument(block, instr, 1);

                block->program[i].op    = MAP;  // TAC
                block->program[i].oper  = EXP;
                block->program[i].out   = out;
                block->program[i].in1   = in1;
                block->program[i].in2   = in2;
            
                block->omask |= MAP;    // Operationmask
                break;
            case BH_EXP2:
                in2 = add_argument(block, instr, 1);

                block->program[i].op    = MAP;  // TAC
                block->program[i].oper  = EXP2;
                block->program[i].out   = out;
                block->program[i].in1   = in1;
                block->program[i].in2   = in2;
            
                block->omask |= MAP;    // Operationmask
                break;
            case BH_EXPM1:
                in2 = add_argument(block, instr, 1);

                block->program[i].op    = MAP;  // TAC
                block->program[i].oper  = EXPM1;
                block->program[i].out   = out;
                block->program[i].in1   = in1;
                block->program[i].in2   = in2;
            
                block->omask |= MAP;    // Operationmask
                break;
            case BH_FLOOR:
                in2 = add_argument(block, instr, 1);

                block->program[i].op    = MAP;  // TAC
                block->program[i].oper  = FLOOR;
                block->program[i].out   = out;
                block->program[i].in1   = in1;
                block->program[i].in2   = in2;
            
                block->omask |= MAP;    // Operationmask
                break;
            case BH_IDENTITY:
                in2 = add_argument(block, instr, 1);

                block->program[i].op    = MAP;  // TAC
                block->program[i].oper  = IDENTITY;
                block->program[i].out   = out;
                block->program[i].in1   = in1;
                block->program[i].in2   = in2;
            
                block->omask |= MAP;    // Operationmask
                break;
            case BH_IMAG:
                in2 = add_argument(block, instr, 1);

                block->program[i].op    = MAP;  // TAC
                block->program[i].oper  = IMAG;
                block->program[i].out   = out;
                block->program[i].in1   = in1;
                block->program[i].in2   = in2;
            
                block->omask |= MAP;    // Operationmask
                break;
            case BH_INVERT:
                in2 = add_argument(block, instr, 1);

                block->program[i].op    = MAP;  // TAC
                block->program[i].oper  = INVERT;
                block->program[i].out   = out;
                block->program[i].in1   = in1;
                block->program[i].in2   = in2;
            
                block->omask |= MAP;    // Operationmask
                break;
            case BH_ISINF:
                in2 = add_argument(block, instr, 1);

                block->program[i].op    = MAP;  // TAC
                block->program[i].oper  = ISINF;
                block->program[i].out   = out;
                block->program[i].in1   = in1;
                block->program[i].in2   = in2;
            
                block->omask |= MAP;    // Operationmask
                break;
            case BH_ISNAN:
                in2 = add_argument(block, instr, 1);

                block->program[i].op    = MAP;  // TAC
                block->program[i].oper  = ISNAN;
                block->program[i].out   = out;
                block->program[i].in1   = in1;
                block->program[i].in2   = in2;
            
                block->omask |= MAP;    // Operationmask
                break;
            case BH_LOG:
                in2 = add_argument(block, instr, 1);

                block->program[i].op    = MAP;  // TAC
                block->program[i].oper  = LOG;
                block->program[i].out   = out;
                block->program[i].in1   = in1;
                block->program[i].in2   = in2;
            
                block->omask |= MAP;    // Operationmask
                break;
            case BH_LOG10:
                in2 = add_argument(block, instr, 1);

                block->program[i].op    = MAP;  // TAC
                block->program[i].oper  = LOG10;
                block->program[i].out   = out;
                block->program[i].in1   = in1;
                block->program[i].in2   = in2;
            
                block->omask |= MAP;    // Operationmask
                break;
            case BH_LOG1P:
                in2 = add_argument(block, instr, 1);

                block->program[i].op    = MAP;  // TAC
                block->program[i].oper  = LOG1P;
                block->program[i].out   = out;
                block->program[i].in1   = in1;
                block->program[i].in2   = in2;
            
                block->omask |= MAP;    // Operationmask
                break;
            case BH_LOG2:
                in2 = add_argument(block, instr, 1);

                block->program[i].op    = MAP;  // TAC
                block->program[i].oper  = LOG2;
                block->program[i].out   = out;
                block->program[i].in1   = in1;
                block->program[i].in2   = in2;
            
                block->omask |= MAP;    // Operationmask
                break;
            case BH_LOGICAL_NOT:
                in2 = add_argument(block, instr, 1);

                block->program[i].op    = MAP;  // TAC
                block->program[i].oper  = LOGICAL_NOT;
                block->program[i].out   = out;
                block->program[i].in1   = in1;
                block->program[i].in2   = in2;
            
                block->omask |= MAP;    // Operationmask
                break;
            case BH_REAL:
                in2 = add_argument(block, instr, 1);

                block->program[i].op    = MAP;  // TAC
                block->program[i].oper  = REAL;
                block->program[i].out   = out;
                block->program[i].in1   = in1;
                block->program[i].in2   = in2;
            
                block->omask |= MAP;    // Operationmask
                break;
            case BH_RINT:
                in2 = add_argument(block, instr, 1);

                block->program[i].op    = MAP;  // TAC
                block->program[i].oper  = RINT;
                block->program[i].out   = out;
                block->program[i].in1   = in1;
                block->program[i].in2   = in2;
            
                block->omask |= MAP;    // Operationmask
                break;
            case BH_SIN:
                in2 = add_argument(block, instr, 1);

                block->program[i].op    = MAP;  // TAC
                block->program[i].oper  = SIN;
                block->program[i].out   = out;
                block->program[i].in1   = in1;
                block->program[i].in2   = in2;
            
                block->omask |= MAP;    // Operationmask
                break;
            case BH_SINH:
                in2 = add_argument(block, instr, 1);

                block->program[i].op    = MAP;  // TAC
                block->program[i].oper  = SINH;
                block->program[i].out   = out;
                block->program[i].in1   = in1;
                block->program[i].in2   = in2;
            
                block->omask |= MAP;    // Operationmask
                break;
            case BH_SQRT:
                in2 = add_argument(block, instr, 1);

                block->program[i].op    = MAP;  // TAC
                block->program[i].oper  = SQRT;
                block->program[i].out   = out;
                block->program[i].in1   = in1;
                block->program[i].in2   = in2;
            
                block->omask |= MAP;    // Operationmask
                break;
            case BH_TAN:
                in2 = add_argument(block, instr, 1);

                block->program[i].op    = MAP;  // TAC
                block->program[i].oper  = TAN;
                block->program[i].out   = out;
                block->program[i].in1   = in1;
                block->program[i].in2   = in2;
            
                block->omask |= MAP;    // Operationmask
                break;
            case BH_TANH:
                in2 = add_argument(block, instr, 1);

                block->program[i].op    = MAP;  // TAC
                block->program[i].oper  = TANH;
                block->program[i].out   = out;
                block->program[i].in1   = in1;
                block->program[i].in2   = in2;
            
                block->omask |= MAP;    // Operationmask
                break;
            case BH_TRUNC:
                in2 = add_argument(block, instr, 1);

                block->program[i].op    = MAP;  // TAC
                block->program[i].oper  = TRUNC;
                block->program[i].out   = out;
                block->program[i].in1   = in1;
                block->program[i].in2   = in2;
            
                block->omask |= MAP;    // Operationmask
                break;
            case BH_ADD:
                in2 = add_argument(block, instr, 1);
                in2 = add_argument(block, instr, 2);

                block->program[i].op    = ZIP;  // TAC
                block->program[i].oper  = ADD;
                block->program[i].out   = out;
                block->program[i].in1   = in1;
                block->program[i].in2   = in2;
            
                block->omask |= ZIP;    // Operationmask
                break;
            case BH_ARCTAN2:
                in2 = add_argument(block, instr, 1);
                in2 = add_argument(block, instr, 2);

                block->program[i].op    = ZIP;  // TAC
                block->program[i].oper  = ARCTAN2;
                block->program[i].out   = out;
                block->program[i].in1   = in1;
                block->program[i].in2   = in2;
            
                block->omask |= ZIP;    // Operationmask
                break;
            case BH_BITWISE_AND:
                in2 = add_argument(block, instr, 1);
                in2 = add_argument(block, instr, 2);

                block->program[i].op    = ZIP;  // TAC
                block->program[i].oper  = BITWISE_AND;
                block->program[i].out   = out;
                block->program[i].in1   = in1;
                block->program[i].in2   = in2;
            
                block->omask |= ZIP;    // Operationmask
                break;
            case BH_BITWISE_OR:
                in2 = add_argument(block, instr, 1);
                in2 = add_argument(block, instr, 2);

                block->program[i].op    = ZIP;  // TAC
                block->program[i].oper  = BITWISE_OR;
                block->program[i].out   = out;
                block->program[i].in1   = in1;
                block->program[i].in2   = in2;
            
                block->omask |= ZIP;    // Operationmask
                break;
            case BH_BITWISE_XOR:
                in2 = add_argument(block, instr, 1);
                in2 = add_argument(block, instr, 2);

                block->program[i].op    = ZIP;  // TAC
                block->program[i].oper  = BITWISE_XOR;
                block->program[i].out   = out;
                block->program[i].in1   = in1;
                block->program[i].in2   = in2;
            
                block->omask |= ZIP;    // Operationmask
                break;
            case BH_DIVIDE:
                in2 = add_argument(block, instr, 1);
                in2 = add_argument(block, instr, 2);

                block->program[i].op    = ZIP;  // TAC
                block->program[i].oper  = DIVIDE;
                block->program[i].out   = out;
                block->program[i].in1   = in1;
                block->program[i].in2   = in2;
            
                block->omask |= ZIP;    // Operationmask
                break;
            case BH_EQUAL:
                in2 = add_argument(block, instr, 1);
                in2 = add_argument(block, instr, 2);

                block->program[i].op    = ZIP;  // TAC
                block->program[i].oper  = EQUAL;
                block->program[i].out   = out;
                block->program[i].in1   = in1;
                block->program[i].in2   = in2;
            
                block->omask |= ZIP;    // Operationmask
                break;
            case BH_GREATER:
                in2 = add_argument(block, instr, 1);
                in2 = add_argument(block, instr, 2);

                block->program[i].op    = ZIP;  // TAC
                block->program[i].oper  = GREATER;
                block->program[i].out   = out;
                block->program[i].in1   = in1;
                block->program[i].in2   = in2;
            
                block->omask |= ZIP;    // Operationmask
                break;
            case BH_GREATER_EQUAL:
                in2 = add_argument(block, instr, 1);
                in2 = add_argument(block, instr, 2);

                block->program[i].op    = ZIP;  // TAC
                block->program[i].oper  = GREATER_EQUAL;
                block->program[i].out   = out;
                block->program[i].in1   = in1;
                block->program[i].in2   = in2;
            
                block->omask |= ZIP;    // Operationmask
                break;
            case BH_LEFT_SHIFT:
                in2 = add_argument(block, instr, 1);
                in2 = add_argument(block, instr, 2);

                block->program[i].op    = ZIP;  // TAC
                block->program[i].oper  = LEFT_SHIFT;
                block->program[i].out   = out;
                block->program[i].in1   = in1;
                block->program[i].in2   = in2;
            
                block->omask |= ZIP;    // Operationmask
                break;
            case BH_LESS:
                in2 = add_argument(block, instr, 1);
                in2 = add_argument(block, instr, 2);

                block->program[i].op    = ZIP;  // TAC
                block->program[i].oper  = LESS;
                block->program[i].out   = out;
                block->program[i].in1   = in1;
                block->program[i].in2   = in2;
            
                block->omask |= ZIP;    // Operationmask
                break;
            case BH_LESS_EQUAL:
                in2 = add_argument(block, instr, 1);
                in2 = add_argument(block, instr, 2);

                block->program[i].op    = ZIP;  // TAC
                block->program[i].oper  = LESS_EQUAL;
                block->program[i].out   = out;
                block->program[i].in1   = in1;
                block->program[i].in2   = in2;
            
                block->omask |= ZIP;    // Operationmask
                break;
            case BH_LOGICAL_AND:
                in2 = add_argument(block, instr, 1);
                in2 = add_argument(block, instr, 2);

                block->program[i].op    = ZIP;  // TAC
                block->program[i].oper  = LOGICAL_AND;
                block->program[i].out   = out;
                block->program[i].in1   = in1;
                block->program[i].in2   = in2;
            
                block->omask |= ZIP;    // Operationmask
                break;
            case BH_LOGICAL_OR:
                in2 = add_argument(block, instr, 1);
                in2 = add_argument(block, instr, 2);

                block->program[i].op    = ZIP;  // TAC
                block->program[i].oper  = LOGICAL_OR;
                block->program[i].out   = out;
                block->program[i].in1   = in1;
                block->program[i].in2   = in2;
            
                block->omask |= ZIP;    // Operationmask
                break;
            case BH_LOGICAL_XOR:
                in2 = add_argument(block, instr, 1);
                in2 = add_argument(block, instr, 2);

                block->program[i].op    = ZIP;  // TAC
                block->program[i].oper  = LOGICAL_XOR;
                block->program[i].out   = out;
                block->program[i].in1   = in1;
                block->program[i].in2   = in2;
            
                block->omask |= ZIP;    // Operationmask
                break;
            case BH_MAXIMUM:
                in2 = add_argument(block, instr, 1);
                in2 = add_argument(block, instr, 2);

                block->program[i].op    = ZIP;  // TAC
                block->program[i].oper  = MAXIMUM;
                block->program[i].out   = out;
                block->program[i].in1   = in1;
                block->program[i].in2   = in2;
            
                block->omask |= ZIP;    // Operationmask
                break;
            case BH_MINIMUM:
                in2 = add_argument(block, instr, 1);
                in2 = add_argument(block, instr, 2);

                block->program[i].op    = ZIP;  // TAC
                block->program[i].oper  = MINIMUM;
                block->program[i].out   = out;
                block->program[i].in1   = in1;
                block->program[i].in2   = in2;
            
                block->omask |= ZIP;    // Operationmask
                break;
            case BH_MOD:
                in2 = add_argument(block, instr, 1);
                in2 = add_argument(block, instr, 2);

                block->program[i].op    = ZIP;  // TAC
                block->program[i].oper  = MOD;
                block->program[i].out   = out;
                block->program[i].in1   = in1;
                block->program[i].in2   = in2;
            
                block->omask |= ZIP;    // Operationmask
                break;
            case BH_MULTIPLY:
                in2 = add_argument(block, instr, 1);
                in2 = add_argument(block, instr, 2);

                block->program[i].op    = ZIP;  // TAC
                block->program[i].oper  = MULTIPLY;
                block->program[i].out   = out;
                block->program[i].in1   = in1;
                block->program[i].in2   = in2;
            
                block->omask |= ZIP;    // Operationmask
                break;
            case BH_NOT_EQUAL:
                in2 = add_argument(block, instr, 1);
                in2 = add_argument(block, instr, 2);

                block->program[i].op    = ZIP;  // TAC
                block->program[i].oper  = NOT_EQUAL;
                block->program[i].out   = out;
                block->program[i].in1   = in1;
                block->program[i].in2   = in2;
            
                block->omask |= ZIP;    // Operationmask
                break;
            case BH_POWER:
                in2 = add_argument(block, instr, 1);
                in2 = add_argument(block, instr, 2);

                block->program[i].op    = ZIP;  // TAC
                block->program[i].oper  = POWER;
                block->program[i].out   = out;
                block->program[i].in1   = in1;
                block->program[i].in2   = in2;
            
                block->omask |= ZIP;    // Operationmask
                break;
            case BH_RIGHT_SHIFT:
                in2 = add_argument(block, instr, 1);
                in2 = add_argument(block, instr, 2);

                block->program[i].op    = ZIP;  // TAC
                block->program[i].oper  = RIGHT_SHIFT;
                block->program[i].out   = out;
                block->program[i].in1   = in1;
                block->program[i].in2   = in2;
            
                block->omask |= ZIP;    // Operationmask
                break;
            case BH_SUBTRACT:
                in2 = add_argument(block, instr, 1);
                in2 = add_argument(block, instr, 2);

                block->program[i].op    = ZIP;  // TAC
                block->program[i].oper  = SUBTRACT;
                block->program[i].out   = out;
                block->program[i].in1   = in1;
                block->program[i].in2   = in2;
            
                block->omask |= ZIP;    // Operationmask
                break;
            case BH_ADD_REDUCE:
                in2 = add_argument(block, instr, 1);
                in2 = add_argument(block, instr, 2);

                block->program[i].op    = REDUCE;  // TAC
                block->program[i].oper  = ADD;
                block->program[i].out   = out;
                block->program[i].in1   = in1;
                block->program[i].in2   = in2;
            
                block->omask |= REDUCE;    // Operationmask
                break;
            case BH_BITWISE_AND_REDUCE:
                in2 = add_argument(block, instr, 1);
                in2 = add_argument(block, instr, 2);

                block->program[i].op    = REDUCE;  // TAC
                block->program[i].oper  = BITWISE_AND;
                block->program[i].out   = out;
                block->program[i].in1   = in1;
                block->program[i].in2   = in2;
            
                block->omask |= REDUCE;    // Operationmask
                break;
            case BH_BITWISE_OR_REDUCE:
                in2 = add_argument(block, instr, 1);
                in2 = add_argument(block, instr, 2);

                block->program[i].op    = REDUCE;  // TAC
                block->program[i].oper  = BITWISE_OR;
                block->program[i].out   = out;
                block->program[i].in1   = in1;
                block->program[i].in2   = in2;
            
                block->omask |= REDUCE;    // Operationmask
                break;
            case BH_BITWISE_XOR_REDUCE:
                in2 = add_argument(block, instr, 1);
                in2 = add_argument(block, instr, 2);

                block->program[i].op    = REDUCE;  // TAC
                block->program[i].oper  = BITWISE_XOR;
                block->program[i].out   = out;
                block->program[i].in1   = in1;
                block->program[i].in2   = in2;
            
                block->omask |= REDUCE;    // Operationmask
                break;
            case BH_LOGICAL_AND_REDUCE:
                in2 = add_argument(block, instr, 1);
                in2 = add_argument(block, instr, 2);

                block->program[i].op    = REDUCE;  // TAC
                block->program[i].oper  = LOGICAL_AND;
                block->program[i].out   = out;
                block->program[i].in1   = in1;
                block->program[i].in2   = in2;
            
                block->omask |= REDUCE;    // Operationmask
                break;
            case BH_LOGICAL_OR_REDUCE:
                in2 = add_argument(block, instr, 1);
                in2 = add_argument(block, instr, 2);

                block->program[i].op    = REDUCE;  // TAC
                block->program[i].oper  = LOGICAL_OR;
                block->program[i].out   = out;
                block->program[i].in1   = in1;
                block->program[i].in2   = in2;
            
                block->omask |= REDUCE;    // Operationmask
                break;
            case BH_LOGICAL_XOR_REDUCE:
                in2 = add_argument(block, instr, 1);
                in2 = add_argument(block, instr, 2);

                block->program[i].op    = REDUCE;  // TAC
                block->program[i].oper  = LOGICAL_XOR;
                block->program[i].out   = out;
                block->program[i].in1   = in1;
                block->program[i].in2   = in2;
            
                block->omask |= REDUCE;    // Operationmask
                break;
            case BH_MAXIMUM_REDUCE:
                in2 = add_argument(block, instr, 1);
                in2 = add_argument(block, instr, 2);

                block->program[i].op    = REDUCE;  // TAC
                block->program[i].oper  = MAXIMUM;
                block->program[i].out   = out;
                block->program[i].in1   = in1;
                block->program[i].in2   = in2;
            
                block->omask |= REDUCE;    // Operationmask
                break;
            case BH_MINIMUM_REDUCE:
                in2 = add_argument(block, instr, 1);
                in2 = add_argument(block, instr, 2);

                block->program[i].op    = REDUCE;  // TAC
                block->program[i].oper  = MINIMUM;
                block->program[i].out   = out;
                block->program[i].in1   = in1;
                block->program[i].in2   = in2;
            
                block->omask |= REDUCE;    // Operationmask
                break;
            case BH_MULTIPLY_REDUCE:
                in2 = add_argument(block, instr, 1);
                in2 = add_argument(block, instr, 2);

                block->program[i].op    = REDUCE;  // TAC
                block->program[i].oper  = MULTIPLY;
                block->program[i].out   = out;
                block->program[i].in1   = in1;
                block->program[i].in2   = in2;
            
                block->omask |= REDUCE;    // Operationmask
                break;
            case BH_ADD_ACCUMULATE:
                in2 = add_argument(block, instr, 1);
                in2 = add_argument(block, instr, 2);

                block->program[i].op    = SCAN;  // TAC
                block->program[i].oper  = ADD;
                block->program[i].out   = out;
                block->program[i].in1   = in1;
                block->program[i].in2   = in2;
            
                block->omask |= SCAN;    // Operationmask
                break;
            case BH_MULTIPLY_ACCUMULATE:
                in2 = add_argument(block, instr, 1);
                in2 = add_argument(block, instr, 2);

                block->program[i].op    = SCAN;  // TAC
                block->program[i].oper  = MULTIPLY;
                block->program[i].out   = out;
                block->program[i].in1   = in1;
                block->program[i].in2   = in2;
            
                block->omask |= SCAN;    // Operationmask
                break;
            case BH_RANDOM:
                // This one requires special-handling... what a beaty...
                in1 = (block->nargs)++;                // Input
                block->scope[in1].layout    = CONSTANT;
                block->scope[in1].data      = &(instr->constant.value.r123.start);
                block->scope[in1].type      = BH_UINT64;
                block->scope[in1].nelem     = 1;

                in2 = (block->nargs)++;
                block->scope[in2].layout    = CONSTANT;
                block->scope[in2].data      = &(instr->constant.value.r123.key);
                block->scope[in2].type      = BH_UINT64;
                block->scope[in2].nelem     = 1;

                block->program[i].op    = GENERATE;  // TAC
                block->program[i].oper  = RANDOM;
                block->program[i].out   = out;
                block->program[i].in1   = in1;
                block->program[i].in2   = in2;
            
                block->omask |= GENERATE;    // Operationmask
                break;
            case BH_RANGE:

                block->program[i].op    = GENERATE;  // TAC
                block->program[i].oper  = RANGE;
                block->program[i].out   = out;
                block->program[i].in1   = in1;
                block->program[i].in2   = in2;
            
                block->omask |= GENERATE;    // Operationmask
                break;
            case BH_DISCARD:
                in2 = add_argument(block, instr, 1);

                block->program[i].op    = SYSTEM;  // TAC
                block->program[i].oper  = DISCARD;
                block->program[i].out   = out;
                block->program[i].in1   = in1;
                block->program[i].in2   = in2;
            
                block->omask |= SYSTEM;    // Operationmask
                break;
            case BH_FREE:

                block->program[i].op    = SYSTEM;  // TAC
                block->program[i].oper  = FREE;
                block->program[i].out   = out;
                block->program[i].in1   = in1;
                block->program[i].in2   = in2;
            
                block->omask |= SYSTEM;    // Operationmask
                break;
            case BH_NONE:
                in2 = add_argument(block, instr, 1);

                block->program[i].op    = SYSTEM;  // TAC
                block->program[i].oper  = NONE;
                block->program[i].out   = out;
                block->program[i].in1   = in1;
                block->program[i].in2   = in2;
            
                block->omask |= SYSTEM;    // Operationmask
                break;
            case BH_SYNC:
                in2 = add_argument(block, instr, 1);

                block->program[i].op    = SYSTEM;  // TAC
                block->program[i].oper  = SYNC;
                block->program[i].out   = out;
                block->program[i].in1   = in1;
                block->program[i].in2   = in2;
            
                block->omask |= SYSTEM;    // Operationmask
                break;

            default:
                if (instr->opcode>=BH_MAX_OPCODE_ID) {   // Handle extensions here
                    block->program[i].op   = EXTENSION; // TODO: Be clever about it
                    block->program[i].oper = EXT_OFFSET;
                    block->program[i].out  = 0;
                    block->program[i].in1  = 0;
                    block->program[i].in2  = 0;

                    printf("Extension-method.\n");
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

void decompose(block_t* block)
{
    free(block->program);
    free(block->scope);
}
