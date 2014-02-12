#compiler-settings
directiveStartToken= %
#end compiler-settings
%slurp

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
    if (bh_is_constant(&instr->operand[operand_idx])) {
        kernel->scope[arg_idx].layout    = CONSTANT;
        kernel->scope[arg_idx].data      = &(instr->constant.value);
        kernel->scope[arg_idx].type      = bh_base_array(&instr->operand[operand_idx])->type;
        kernel->scope[arg_idx].nelem     = 1;
    } else {
        if (is_contiguous(&kernel->scope[arg_idx])) {
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
    kernel->program = (tac_t*)malloc(dag->nnode*sizeof(tac_t));

    for (int i=0; i<dag->nnode; ++i) {
        bh_instruction* instr = kernel->instr[i] = &ir->instr_list[dag->node_map[i]];
        kernel->tsig[i]  = bh_type_sig(instr);
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

            %for $opcode, $operation, $operator, $nin in $operations
            case $opcode:
                %if $opcode == 'BH_RANDOM'
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
                %else
                %if nin >= 1
                in2 = add_argument(kernel, instr, 1);
                %end if
                %if nin >= 2
                in2 = add_argument(kernel, instr, 2);
                %end if
                %end if

                kernel->program[i].op    = $operation;  // TAC
                kernel->program[i].oper  = $operator;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;
            
                kernel->omask |= $operation;    // Operationmask
                break;
            %end for

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
