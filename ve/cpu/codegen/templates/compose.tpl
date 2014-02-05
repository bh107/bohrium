#compiler-settings
directiveStartToken= %
#end compiler-settings
%slurp

/**
 *  Compose a kernel based on a bunch of bh_instructions.
 */
static bh_error compose(bh_kernel_t* kernel, bh_ir* ir, bh_dag* dag)
{
    kernel->nargs   = 0;
    kernel->args    = (bh_kernel_arg_t*)malloc(3*dag->nnode*sizeof(bh_kernel_arg_t));
    kernel->program = (bytecode_t*)malloc(dag->nnode*sizeof(bytecode_t));

    for (int i=0; i<dag->nnode; ++i) {
        bh_instruction* instr = &ir->instr_list[dag->node_map[i]];
        int lmask = bh_layoutmask(instr);
        kernel->lmask[i] = lmask;

        int out=0, in1=0, in2=0;

        // All but BH_NONE has an output which is an array
        if (instr->opcode != BH_NONE) {
            out = (kernel->nargs)++;

            kernel->args[out].data      = bh_base_array(&instr->operand[0])->data;
            kernel->args[out].type      = bh_base_array(&instr->operand[0])->type;
            kernel->args[out].nelem     = bh_base_array(&instr->operand[0])->nelem;
            kernel->args[out].ndim      = instr->operand[0].ndim;
            kernel->args[out].start     = instr->operand[0].start;
            kernel->args[out].shape     = instr->operand[0].shape;
            kernel->args[out].stride    = instr->operand[0].stride;
        }

        //
        // Program packing
        switch (instr->opcode) {    // [OPCODE_SWITCH]

            // System operation
            %for $opcode, $operation, $operator in $system
            case $opcode:
                //kernel->program[i] = {$operation, $operator, out, in1, in2};
                // Setup bytecode
                kernel->program[i].op    = $operation;
                kernel->program[i].oper  = $operator;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;
                break;
            %end for

            //
            // Reduce operation with binary operator
            //
            %for $opcode, $operation, $operator in $reductions
            case $opcode:
                in1 = (kernel->nargs)++;
                in2 = (kernel->nargs)++;                

                kernel->args[in1].data      = bh_base_array(&instr->operand[1])->data;
                kernel->args[in1].type      = bh_base_array(&instr->operand[1])->type;
                kernel->args[in1].nelem     = bh_base_array(&instr->operand[1])->nelem;
                kernel->args[in1].ndim      = instr->operand[1].ndim;
                kernel->args[in1].start     = instr->operand[1].start;
                kernel->args[in1].shape     = instr->operand[1].shape;
                kernel->args[in1].stride    = instr->operand[1].stride;

                kernel->args[in2].data = &(instr->constant.value);
                kernel->args[in2].type = bh_base_array(&instr->operand[2])->type;

                //kernel->program[i] = {$operation, $operator, out, in1, in2};
                // Setup bytecode
                kernel->program[i].op    = $operation;
                kernel->program[i].oper  = $operator;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;
                break;
            %end for

            //
            // Scan operation with binary operator
            //
            %for $opcode, $operation, $operator in $scans
            case $opcode:
                in1 = (kernel->nargs)++;
                in2 = (kernel->nargs)++;                

                kernel->args[in1].data      = bh_base_array(&instr->operand[1])->data;
                kernel->args[in1].type      = bh_base_array(&instr->operand[1])->type;
                kernel->args[in1].nelem     = bh_base_array(&instr->operand[1])->nelem;
                kernel->args[in1].ndim      = instr->operand[1].ndim;
                kernel->args[in1].start     = instr->operand[1].start;
                kernel->args[in1].shape     = instr->operand[1].shape;
                kernel->args[in1].stride    = instr->operand[1].stride;

                kernel->args[in2].data = &(instr->constant.value);
                kernel->args[in2].type = bh_base_array(&instr->operand[2])->type;

                //kernel->program[i] = {$operation, $operator, out, in1, in2};
                // Setup bytecode
                kernel->program[i].op    = $operation;
                kernel->program[i].oper  = $operator;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;
                break;
            %end for

            //
            // Elementwise operation with unary operator
            //
            %for $opcode, $operation, $operator in $ewise_u
            case $opcode:
                in1 = (kernel->nargs)++;

                if ((lmask & A1_CONSTANT) == A1_CONSTANT) {
                    kernel->args[in1].data = &(instr->constant.value);
                    kernel->args[in1].type = bh_base_array(&instr->operand[1])->type;
                } else {
                    kernel->args[in1].data   = bh_base_array(&instr->operand[1])->data;
                    kernel->args[in1].type = bh_base_array(&instr->operand[1])->type;
                    kernel->args[in1].nelem  = bh_base_array(&instr->operand[1])->nelem;
                    kernel->args[in1].ndim   = instr->operand[1].ndim;
                    kernel->args[in1].start  = instr->operand[1].start;
                    kernel->args[in1].shape  = instr->operand[1].shape;
                    kernel->args[in1].stride = instr->operand[1].stride;
                }

                //kernel->program[i] = {$operation, $operator, out, in1, 0};
                // Setup bytecode
                kernel->program[i].op    = $operation;
                kernel->program[i].oper  = $operator;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = 0;
                break;
            %end for

            //
            // Elementwise operation with binary operator
            //
            %for $opcode, $operation, $operator in $ewise_b
            case $opcode:
                in1 = (kernel->nargs)++;
                in2 = (kernel->nargs)++;

                if ((lmask & A2_CONSTANT) == A2_CONSTANT) {         // AAK
                    kernel->args[in1].data   = bh_base_array(&instr->operand[1])->data;
                    kernel->args[in1].type   = bh_base_array(&instr->operand[1])->type;
                    kernel->args[in1].nelem  = bh_base_array(&instr->operand[1])->nelem;
                    kernel->args[in1].ndim   = instr->operand[1].ndim;
                    kernel->args[in1].start  = instr->operand[1].start;
                    kernel->args[in1].shape  = instr->operand[1].shape;
                    kernel->args[in1].stride = instr->operand[1].stride;

                    kernel->args[in2].data = &(instr->constant.value);
                    kernel->args[in2].type   = bh_base_array(&instr->operand[2])->type;
                } else if ((lmask & A1_CONSTANT) == A1_CONSTANT) {  // AKA
                    kernel->args[in1].data = &(instr->constant.value);
                    kernel->args[in1].type   = bh_base_array(&instr->operand[1])->type;

                    kernel->args[in2].data   = bh_base_array(&instr->operand[2])->data;
                    kernel->args[in2].type   = bh_base_array(&instr->operand[2])->type;
                    kernel->args[in2].nelem  = bh_base_array(&instr->operand[2])->nelem;
                    kernel->args[in2].ndim   = instr->operand[2].ndim;
                    kernel->args[in2].start  = instr->operand[2].start;
                    kernel->args[in2].shape  = instr->operand[2].shape;
                    kernel->args[in2].stride = instr->operand[2].stride;
                } else {                                            // AAA
                    kernel->args[in1].data   = bh_base_array(&instr->operand[1])->data;
                    kernel->args[in1].type   = bh_base_array(&instr->operand[1])->type;
                    kernel->args[in1].nelem  = bh_base_array(&instr->operand[1])->nelem;
                    kernel->args[in1].ndim   = instr->operand[1].ndim;
                    kernel->args[in1].start  = instr->operand[1].start;
                    kernel->args[in1].shape  = instr->operand[1].shape;
                    kernel->args[in1].stride = instr->operand[1].stride;

                    kernel->args[in2].data   = bh_base_array(&instr->operand[2])->data;
                    kernel->args[in2].type   = bh_base_array(&instr->operand[2])->type;
                    kernel->args[in2].nelem  = bh_base_array(&instr->operand[2])->nelem;
                    kernel->args[in2].ndim   = instr->operand[2].ndim;
                    kernel->args[in2].start  = instr->operand[2].start;
                    kernel->args[in2].shape  = instr->operand[2].shape;
                    kernel->args[in2].stride = instr->operand[2].stride;
                }

                //kernel->program[i] = {$operation, $operator, out, in1, in2};
                // Setup bytecode
                kernel->program[i].op    = $operation;
                kernel->program[i].oper  = $operator;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;
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
