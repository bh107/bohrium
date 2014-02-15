#compiler-settings
directiveStartToken= %
#end compiler-settings
%slurp

/**
 *  Add instruction operand as argument to block.
 *
 *  @param instr        The instruction whos operand should be converted.
 *  @param operand_idx  Index of the operand to represent as arg_t
 *  @param block        The block in which scope the argument will exist.
 */
static uint32_t add_argument(block_t* block, bh_instruction* instr, int operand_idx)
{
    uint32_t arg_idx = ++(block->nargs);
    if (bh_is_constant(&instr->operand[operand_idx])) {
        block->scope[arg_idx].layout    = CONSTANT;
        block->scope[arg_idx].data      = &(instr->constant.value);
        block->scope[arg_idx].type      = instr->constant.type;
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
    block->nargs    = 0;
    block->omask    = 0;
    block->scope    = (block_arg_t*)malloc(1+3*dag->nnode*sizeof(block_arg_t));
    block->program  = (tac_t*)malloc(dag->nnode*sizeof(tac_t));
    block->length   = dag->nnode;

    for (int i=0; i<dag->nnode; ++i) {
        block->instr[i] = &ir->instr_list[dag->node_map[i]];
        bh_instruction* instr = block->instr[i];

        uint32_t out=0, in1=0, in2=0;

        //
        // Program packing: output argument
        // NOTE: All but BH_NONE has an output which is an array
        if (instr->opcode != BH_NONE) {
            out = add_argument(block, instr, 0);
        }

        //
        // Program packing; operator, operand and input argument(s).
        switch (instr->opcode) {    // [OPCODE_SWITCH]

            %for $opcode, $operation, $operator, $nin in $operations
            case $opcode:
                %if $opcode == 'BH_RANDOM'
                // This one requires special-handling... what a beaty...
                in1 = ++(block->nargs);                // Input
                block->scope[in1].layout    = CONSTANT;
                block->scope[in1].data      = &(instr->constant.value.r123.start);
                block->scope[in1].type      = BH_UINT64;
                block->scope[in1].nelem     = 1;

                in2 = ++(block->nargs);
                block->scope[in2].layout    = CONSTANT;
                block->scope[in2].data      = &(instr->constant.value.r123.key);
                block->scope[in2].type      = BH_UINT64;
                block->scope[in2].nelem     = 1;
                %else if 'ACCUMULATE' in $opcode or 'REDUCE' in $opcode
                in1 = add_argument(block, instr, 1);

                in2 = ++(block->nargs);
                block->scope[in2].layout    = CONSTANT;
                block->scope[in2].data      = &(instr->constant.value.r123.key);
                block->scope[in2].type      = BH_UINT64;
                block->scope[in2].nelem     = 1;
                %else
                %if nin >= 1
                in1 = add_argument(block, instr, 1);
                %end if
                %if nin >= 2
                in2 = add_argument(block, instr, 2);
                %end if
                %end if

                block->program[i].op    = $operation;  // TAC
                block->program[i].oper  = $operator;
                block->program[i].out   = out;
                block->program[i].in1   = in1;
                block->program[i].in2   = in2;
            
                block->omask |= $operation;    // Operationmask
                break;
            %end for

            default:
                if (instr->opcode>=BH_MAX_OPCODE_ID) {   // Handle extensions here

                    block->program[i].op   = EXTENSION; // TODO: Be clever about it
                    block->program[i].oper = EXT_OFFSET;
                    block->program[i].out  = 0;
                    block->program[i].in1  = 0;
                    block->program[i].in2  = 0;

                    cout << "Extension method." << endl;
                } else {
                    in1 = 1;
                    in2 = 2;
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
