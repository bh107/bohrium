
/**
 *  Compose a kernel based on a bunch of bh_instructions.
 */
static bh_error compose(bh_kernel_t* kernel, bh_ir* ir, bh_dag* dag)
{
    kernel->nargs   = 0;
    kernel->args    = (bh_kernel_arg_t*)malloc(3*dag->nnode*sizeof(bh_kernel_arg_t));
    kernel->program = (bytecode_t*)malloc(dag->nnode*sizeof(bytecode_t));

    for (int i=0; i<dag->nnode; ++i) {
        kernel->tsig[i]  = bh_type_sig(instr);

        bh_instruction* instr = kernel->instr[i] = &ir->instr_list[dag->node_map[i]];
        int lmask = kernel->lmask[i] = bh_layoutmask(instr);
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
            case BH_DISCARD:            
                // Setup bytecode
                kernel->program[i].op    = SYSTEM;
                kernel->program[i].oper  = DISCARD;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;

                // Update the operationmask
                kernel->omask = kernel->omask | SYSTEM;
                break;
            case BH_FREE:            
                // Setup bytecode
                kernel->program[i].op    = SYSTEM;
                kernel->program[i].oper  = FREE;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;

                // Update the operationmask
                kernel->omask = kernel->omask | SYSTEM;
                break;
            case BH_NONE:            
                // Setup bytecode
                kernel->program[i].op    = SYSTEM;
                kernel->program[i].oper  = NONE;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;

                // Update the operationmask
                kernel->omask = kernel->omask | SYSTEM;
                break;
            case BH_SYNC:            
                // Setup bytecode
                kernel->program[i].op    = SYSTEM;
                kernel->program[i].oper  = SYNC;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;

                // Update the operationmask
                kernel->omask = kernel->omask | SYSTEM;
                break;

            //
            // Reduce operation with binary operator
            //
            case BH_ADD_REDUCE:
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

                // Setup bytecode
                kernel->program[i].op    = REDUCE;
                kernel->program[i].oper  = ADD;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;

                // Update the operationmask
                kernel->omask = kernel->omask | REDUCE;
                break;
            case BH_BITWISE_AND_REDUCE:
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

                // Setup bytecode
                kernel->program[i].op    = REDUCE;
                kernel->program[i].oper  = BITWISE_AND;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;

                // Update the operationmask
                kernel->omask = kernel->omask | REDUCE;
                break;
            case BH_BITWISE_OR_REDUCE:
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

                // Setup bytecode
                kernel->program[i].op    = REDUCE;
                kernel->program[i].oper  = BITWISE_OR;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;

                // Update the operationmask
                kernel->omask = kernel->omask | REDUCE;
                break;
            case BH_BITWISE_XOR_REDUCE:
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

                // Setup bytecode
                kernel->program[i].op    = REDUCE;
                kernel->program[i].oper  = BITWISE_XOR;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;

                // Update the operationmask
                kernel->omask = kernel->omask | REDUCE;
                break;
            case BH_LOGICAL_AND_REDUCE:
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

                // Setup bytecode
                kernel->program[i].op    = REDUCE;
                kernel->program[i].oper  = LOGICAL_AND;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;

                // Update the operationmask
                kernel->omask = kernel->omask | REDUCE;
                break;
            case BH_LOGICAL_OR_REDUCE:
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

                // Setup bytecode
                kernel->program[i].op    = REDUCE;
                kernel->program[i].oper  = LOGICAL_OR;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;

                // Update the operationmask
                kernel->omask = kernel->omask | REDUCE;
                break;
            case BH_LOGICAL_XOR_REDUCE:
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

                // Setup bytecode
                kernel->program[i].op    = REDUCE;
                kernel->program[i].oper  = LOGICAL_XOR;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;

                // Update the operationmask
                kernel->omask = kernel->omask | REDUCE;
                break;
            case BH_MAXIMUM_REDUCE:
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

                // Setup bytecode
                kernel->program[i].op    = REDUCE;
                kernel->program[i].oper  = MAXIMUM;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;

                // Update the operationmask
                kernel->omask = kernel->omask | REDUCE;
                break;
            case BH_MINIMUM_REDUCE:
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

                // Setup bytecode
                kernel->program[i].op    = REDUCE;
                kernel->program[i].oper  = MINIMUM;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;

                // Update the operationmask
                kernel->omask = kernel->omask | REDUCE;
                break;
            case BH_MULTIPLY_REDUCE:
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

                // Setup bytecode
                kernel->program[i].op    = REDUCE;
                kernel->program[i].oper  = MULTIPLY;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;

                // Update the operationmask
                kernel->omask = kernel->omask | REDUCE;
                break;

            //
            // Scan operation with binary operator
            //
            case BH_ADD_ACCUMULATE:
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

                // Setup bytecode
                kernel->program[i].op    = SCAN;
                kernel->program[i].oper  = ADD;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;

                // Update the operationmask
                kernel->omask = kernel->omask | REDUCE;
                break;
            case BH_MULTIPLY_ACCUMULATE:
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

                // Setup bytecode
                kernel->program[i].op    = SCAN;
                kernel->program[i].oper  = MULTIPLY;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;

                // Update the operationmask
                kernel->omask = kernel->omask | REDUCE;
                break;

            //
            // Elementwise operation with unary operator
            //
            case BH_ABSOLUTE:
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

                // Setup bytecode
                kernel->program[i].op    = EWISE_U;
                kernel->program[i].oper  = ABSOLUTE;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = 0;

                // Update the operationmask
                kernel->omask = kernel->omask | EWISE_U;
                break;
            case BH_ARCCOS:
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

                // Setup bytecode
                kernel->program[i].op    = EWISE_U;
                kernel->program[i].oper  = ARCCOS;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = 0;

                // Update the operationmask
                kernel->omask = kernel->omask | EWISE_U;
                break;
            case BH_ARCCOSH:
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

                // Setup bytecode
                kernel->program[i].op    = EWISE_U;
                kernel->program[i].oper  = ARCCOSH;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = 0;

                // Update the operationmask
                kernel->omask = kernel->omask | EWISE_U;
                break;
            case BH_ARCSIN:
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

                // Setup bytecode
                kernel->program[i].op    = EWISE_U;
                kernel->program[i].oper  = ARCSIN;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = 0;

                // Update the operationmask
                kernel->omask = kernel->omask | EWISE_U;
                break;
            case BH_ARCSINH:
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

                // Setup bytecode
                kernel->program[i].op    = EWISE_U;
                kernel->program[i].oper  = ARCSINH;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = 0;

                // Update the operationmask
                kernel->omask = kernel->omask | EWISE_U;
                break;
            case BH_ARCTAN:
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

                // Setup bytecode
                kernel->program[i].op    = EWISE_U;
                kernel->program[i].oper  = ARCTAN;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = 0;

                // Update the operationmask
                kernel->omask = kernel->omask | EWISE_U;
                break;
            case BH_ARCTANH:
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

                // Setup bytecode
                kernel->program[i].op    = EWISE_U;
                kernel->program[i].oper  = ARCTANH;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = 0;

                // Update the operationmask
                kernel->omask = kernel->omask | EWISE_U;
                break;
            case BH_CEIL:
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

                // Setup bytecode
                kernel->program[i].op    = EWISE_U;
                kernel->program[i].oper  = CEIL;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = 0;

                // Update the operationmask
                kernel->omask = kernel->omask | EWISE_U;
                break;
            case BH_COS:
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

                // Setup bytecode
                kernel->program[i].op    = EWISE_U;
                kernel->program[i].oper  = COS;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = 0;

                // Update the operationmask
                kernel->omask = kernel->omask | EWISE_U;
                break;
            case BH_COSH:
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

                // Setup bytecode
                kernel->program[i].op    = EWISE_U;
                kernel->program[i].oper  = COSH;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = 0;

                // Update the operationmask
                kernel->omask = kernel->omask | EWISE_U;
                break;
            case BH_EXP:
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

                // Setup bytecode
                kernel->program[i].op    = EWISE_U;
                kernel->program[i].oper  = EXP;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = 0;

                // Update the operationmask
                kernel->omask = kernel->omask | EWISE_U;
                break;
            case BH_EXP2:
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

                // Setup bytecode
                kernel->program[i].op    = EWISE_U;
                kernel->program[i].oper  = EXP2;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = 0;

                // Update the operationmask
                kernel->omask = kernel->omask | EWISE_U;
                break;
            case BH_EXPM1:
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

                // Setup bytecode
                kernel->program[i].op    = EWISE_U;
                kernel->program[i].oper  = EXPM1;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = 0;

                // Update the operationmask
                kernel->omask = kernel->omask | EWISE_U;
                break;
            case BH_FLOOR:
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

                // Setup bytecode
                kernel->program[i].op    = EWISE_U;
                kernel->program[i].oper  = FLOOR;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = 0;

                // Update the operationmask
                kernel->omask = kernel->omask | EWISE_U;
                break;
            case BH_IDENTITY:
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

                // Setup bytecode
                kernel->program[i].op    = EWISE_U;
                kernel->program[i].oper  = IDENTITY;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = 0;

                // Update the operationmask
                kernel->omask = kernel->omask | EWISE_U;
                break;
            case BH_IMAG:
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

                // Setup bytecode
                kernel->program[i].op    = EWISE_U;
                kernel->program[i].oper  = IMAG;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = 0;

                // Update the operationmask
                kernel->omask = kernel->omask | EWISE_U;
                break;
            case BH_INVERT:
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

                // Setup bytecode
                kernel->program[i].op    = EWISE_U;
                kernel->program[i].oper  = INVERT;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = 0;

                // Update the operationmask
                kernel->omask = kernel->omask | EWISE_U;
                break;
            case BH_ISINF:
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

                // Setup bytecode
                kernel->program[i].op    = EWISE_U;
                kernel->program[i].oper  = ISINF;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = 0;

                // Update the operationmask
                kernel->omask = kernel->omask | EWISE_U;
                break;
            case BH_ISNAN:
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

                // Setup bytecode
                kernel->program[i].op    = EWISE_U;
                kernel->program[i].oper  = ISNAN;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = 0;

                // Update the operationmask
                kernel->omask = kernel->omask | EWISE_U;
                break;
            case BH_LOG:
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

                // Setup bytecode
                kernel->program[i].op    = EWISE_U;
                kernel->program[i].oper  = LOG;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = 0;

                // Update the operationmask
                kernel->omask = kernel->omask | EWISE_U;
                break;
            case BH_LOG10:
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

                // Setup bytecode
                kernel->program[i].op    = EWISE_U;
                kernel->program[i].oper  = LOG10;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = 0;

                // Update the operationmask
                kernel->omask = kernel->omask | EWISE_U;
                break;
            case BH_LOG1P:
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

                // Setup bytecode
                kernel->program[i].op    = EWISE_U;
                kernel->program[i].oper  = LOG1P;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = 0;

                // Update the operationmask
                kernel->omask = kernel->omask | EWISE_U;
                break;
            case BH_LOG2:
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

                // Setup bytecode
                kernel->program[i].op    = EWISE_U;
                kernel->program[i].oper  = LOG2;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = 0;

                // Update the operationmask
                kernel->omask = kernel->omask | EWISE_U;
                break;
            case BH_LOGICAL_NOT:
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

                // Setup bytecode
                kernel->program[i].op    = EWISE_U;
                kernel->program[i].oper  = LOGICAL_NOT;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = 0;

                // Update the operationmask
                kernel->omask = kernel->omask | EWISE_U;
                break;
            case BH_REAL:
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

                // Setup bytecode
                kernel->program[i].op    = EWISE_U;
                kernel->program[i].oper  = REAL;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = 0;

                // Update the operationmask
                kernel->omask = kernel->omask | EWISE_U;
                break;
            case BH_RINT:
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

                // Setup bytecode
                kernel->program[i].op    = EWISE_U;
                kernel->program[i].oper  = RINT;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = 0;

                // Update the operationmask
                kernel->omask = kernel->omask | EWISE_U;
                break;
            case BH_SIN:
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

                // Setup bytecode
                kernel->program[i].op    = EWISE_U;
                kernel->program[i].oper  = SIN;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = 0;

                // Update the operationmask
                kernel->omask = kernel->omask | EWISE_U;
                break;
            case BH_SINH:
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

                // Setup bytecode
                kernel->program[i].op    = EWISE_U;
                kernel->program[i].oper  = SINH;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = 0;

                // Update the operationmask
                kernel->omask = kernel->omask | EWISE_U;
                break;
            case BH_SQRT:
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

                // Setup bytecode
                kernel->program[i].op    = EWISE_U;
                kernel->program[i].oper  = SQRT;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = 0;

                // Update the operationmask
                kernel->omask = kernel->omask | EWISE_U;
                break;
            case BH_TAN:
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

                // Setup bytecode
                kernel->program[i].op    = EWISE_U;
                kernel->program[i].oper  = TAN;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = 0;

                // Update the operationmask
                kernel->omask = kernel->omask | EWISE_U;
                break;
            case BH_TANH:
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

                // Setup bytecode
                kernel->program[i].op    = EWISE_U;
                kernel->program[i].oper  = TANH;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = 0;

                // Update the operationmask
                kernel->omask = kernel->omask | EWISE_U;
                break;
            case BH_TRUNC:
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

                // Setup bytecode
                kernel->program[i].op    = EWISE_U;
                kernel->program[i].oper  = TRUNC;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = 0;

                // Update the operationmask
                kernel->omask = kernel->omask | EWISE_U;
                break;

            //
            // Elementwise operation with binary operator
            //
            case BH_ADD:
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

                // Setup bytecode
                kernel->program[i].op    = EWISE_B;
                kernel->program[i].oper  = ADD;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;

                // Update the operationmask
                kernel->omask = kernel->omask | EWISE_U;
                break;
            case BH_ARCTAN2:
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

                // Setup bytecode
                kernel->program[i].op    = EWISE_B;
                kernel->program[i].oper  = ARCTAN2;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;

                // Update the operationmask
                kernel->omask = kernel->omask | EWISE_U;
                break;
            case BH_BITWISE_AND:
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

                // Setup bytecode
                kernel->program[i].op    = EWISE_B;
                kernel->program[i].oper  = BITWISE_AND;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;

                // Update the operationmask
                kernel->omask = kernel->omask | EWISE_U;
                break;
            case BH_BITWISE_OR:
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

                // Setup bytecode
                kernel->program[i].op    = EWISE_B;
                kernel->program[i].oper  = BITWISE_OR;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;

                // Update the operationmask
                kernel->omask = kernel->omask | EWISE_U;
                break;
            case BH_BITWISE_XOR:
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

                // Setup bytecode
                kernel->program[i].op    = EWISE_B;
                kernel->program[i].oper  = BITWISE_XOR;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;

                // Update the operationmask
                kernel->omask = kernel->omask | EWISE_U;
                break;
            case BH_DIVIDE:
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

                // Setup bytecode
                kernel->program[i].op    = EWISE_B;
                kernel->program[i].oper  = DIVIDE;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;

                // Update the operationmask
                kernel->omask = kernel->omask | EWISE_U;
                break;
            case BH_EQUAL:
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

                // Setup bytecode
                kernel->program[i].op    = EWISE_B;
                kernel->program[i].oper  = EQUAL;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;

                // Update the operationmask
                kernel->omask = kernel->omask | EWISE_U;
                break;
            case BH_GREATER:
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

                // Setup bytecode
                kernel->program[i].op    = EWISE_B;
                kernel->program[i].oper  = GREATER;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;

                // Update the operationmask
                kernel->omask = kernel->omask | EWISE_U;
                break;
            case BH_GREATER_EQUAL:
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

                // Setup bytecode
                kernel->program[i].op    = EWISE_B;
                kernel->program[i].oper  = GREATER_EQUAL;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;

                // Update the operationmask
                kernel->omask = kernel->omask | EWISE_U;
                break;
            case BH_LEFT_SHIFT:
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

                // Setup bytecode
                kernel->program[i].op    = EWISE_B;
                kernel->program[i].oper  = LEFT_SHIFT;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;

                // Update the operationmask
                kernel->omask = kernel->omask | EWISE_U;
                break;
            case BH_LESS:
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

                // Setup bytecode
                kernel->program[i].op    = EWISE_B;
                kernel->program[i].oper  = LESS;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;

                // Update the operationmask
                kernel->omask = kernel->omask | EWISE_U;
                break;
            case BH_LESS_EQUAL:
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

                // Setup bytecode
                kernel->program[i].op    = EWISE_B;
                kernel->program[i].oper  = LESS_EQUAL;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;

                // Update the operationmask
                kernel->omask = kernel->omask | EWISE_U;
                break;
            case BH_LOGICAL_AND:
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

                // Setup bytecode
                kernel->program[i].op    = EWISE_B;
                kernel->program[i].oper  = LOGICAL_AND;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;

                // Update the operationmask
                kernel->omask = kernel->omask | EWISE_U;
                break;
            case BH_LOGICAL_OR:
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

                // Setup bytecode
                kernel->program[i].op    = EWISE_B;
                kernel->program[i].oper  = LOGICAL_OR;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;

                // Update the operationmask
                kernel->omask = kernel->omask | EWISE_U;
                break;
            case BH_LOGICAL_XOR:
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

                // Setup bytecode
                kernel->program[i].op    = EWISE_B;
                kernel->program[i].oper  = LOGICAL_XOR;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;

                // Update the operationmask
                kernel->omask = kernel->omask | EWISE_U;
                break;
            case BH_MAXIMUM:
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

                // Setup bytecode
                kernel->program[i].op    = EWISE_B;
                kernel->program[i].oper  = MAXIMUM;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;

                // Update the operationmask
                kernel->omask = kernel->omask | EWISE_U;
                break;
            case BH_MINIMUM:
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

                // Setup bytecode
                kernel->program[i].op    = EWISE_B;
                kernel->program[i].oper  = MINIMUM;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;

                // Update the operationmask
                kernel->omask = kernel->omask | EWISE_U;
                break;
            case BH_MOD:
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

                // Setup bytecode
                kernel->program[i].op    = EWISE_B;
                kernel->program[i].oper  = MOD;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;

                // Update the operationmask
                kernel->omask = kernel->omask | EWISE_U;
                break;
            case BH_MULTIPLY:
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

                // Setup bytecode
                kernel->program[i].op    = EWISE_B;
                kernel->program[i].oper  = MULTIPLY;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;

                // Update the operationmask
                kernel->omask = kernel->omask | EWISE_U;
                break;
            case BH_NOT_EQUAL:
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

                // Setup bytecode
                kernel->program[i].op    = EWISE_B;
                kernel->program[i].oper  = NOT_EQUAL;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;

                // Update the operationmask
                kernel->omask = kernel->omask | EWISE_U;
                break;
            case BH_POWER:
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

                // Setup bytecode
                kernel->program[i].op    = EWISE_B;
                kernel->program[i].oper  = POWER;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;

                // Update the operationmask
                kernel->omask = kernel->omask | EWISE_U;
                break;
            case BH_RIGHT_SHIFT:
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

                // Setup bytecode
                kernel->program[i].op    = EWISE_B;
                kernel->program[i].oper  = RIGHT_SHIFT;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;

                // Update the operationmask
                kernel->omask = kernel->omask | EWISE_U;
                break;
            case BH_SUBTRACT:
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

                // Setup bytecode
                kernel->program[i].op    = EWISE_B;
                kernel->program[i].oper  = SUBTRACT;
                kernel->program[i].out   = out;
                kernel->program[i].in1   = in1;
                kernel->program[i].in2   = in2;

                // Update the operationmask
                kernel->omask = kernel->omask | EWISE_U;
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
