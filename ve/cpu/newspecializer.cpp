/**
 *  Choose the template.
 *
 *  Contract: Do not call this for system or extension operations.
 */
string template_filename(bh_kernel_t* kernel, int pc, bh_intp optimized)
{
    string tpl_ndim   = "nd.",
           tpl_opcode,
           tpl_layout = "strided.";

    bytecode_t* bytecode = &kernel->program[pc];
    int ndim = (bytecode->op == REDUCE)         ? \
               kernel->args[bytecode->in1].ndim : \
               kernel->args[bytecode->out].ndim;
    int lmask = kernel->lmask[pc];

    switch (bytecode->op) {                    // OPCODE_SWITCH

        case GENERATOR:
            switch(bytecode->oper) {
                case RANDOM:
                    tpl_opcode = "random.";
                    break;
                case RANGE:
                    tpl_opcode = "range.";
                default:
                    printf("Operator x is not supported with operation y\n");
            }
            tpl_layout = "cont.";
            break;

        case SCAN:
            tpl_opcode = "scan.";
            if (optimized && (ndim == 1)) {
                tpl_ndim = "1d.";
            }
            break;

        case REDUCE:
            tpl_opcode = "reduce.";
            if (optimized && (ndim == 1)) {
                tpl_ndim = "1d.";
            } else if (optimized && (ndim == 2)) {
                tpl_ndim = "2d.";
            } else if (optimized && (ndim == 3)) {
                tpl_ndim = "3d.";
            }
            break;

        case EWISE_B:
            
            tpl_opcode  = "ewise.";
            if ((optimized) && ( \
                (lmask == (A0_CONTIGUOUS + A1_CONTIGUOUS + A2_CONTIGUOUS)) || \
                (lmask == (A0_CONTIGUOUS + A1_CONSTANT   + A2_CONTIGUOUS)) || \
                (lmask == (A0_CONTIGUOUS + A1_CONTIGUOUS + A2_CONSTANT)))) {
                tpl_layout  = "cont.";
            } else if ((optimized) && (ndim == 1)) {
                tpl_ndim = "1d.";
            } else if ((optimized) && (ndim == 2)) {
                tpl_ndim = "2d.";
            } else if ((optimized) && (ndim == 3)) {
                tpl_ndim = "3d.";
            }
            break;

        case EWISE_U:

            tpl_opcode  = "ewise.";
            if ((optimized) && ((lmask == (A0_CONTIGUOUS + A1_CONTIGUOUS)) || \
                (lmask == (A0_CONTIGUOUS + A1_CONSTANT)))) {
                tpl_layout  = "cont.";
            } else if ((optimized) && (ndim == 1)) {
                tpl_ndim = "1d.";
            } else if ((optimized) && (ndim == 2)) {
                tpl_ndim = "2d.";
            } else if ((optimized) && (ndim == 3)) {
                tpl_ndim = "3d.";
            }
            break;

        default:
            printf("template_filename: Err=[Unsupported operation %d.]\n", bytecode->oper);
            throw runtime_error("template_filename: No template for opcode.");
    }

    return tpl_opcode + tpl_layout + tpl_ndim  + "tpl";
}


/**
 *  Construct the c-sourcecode for the given kernel.
 *
 *  NOTE: System opcodes are ignored.
 *
 *  @param optimized The level of optimizations to apply to the generated code.
 *  @param kernel The kernel to generate sourcecode for.
 *  @return The generated sourcecode.
 *
 */
string specialize(bh_kernel_t &kernel, bh_intp const optimized) {

    string sourcecode  = "";

    ctemplate::TemplateDictionary kernel_d("KERNEL");   // Kernel - function wrapping code
    kernel_d.SetValue("SYMBOL", kernel.symbol);

    for(int j=0; j<kernel.ninstr; ++j) {
        
        //
        // Grab the instruction for which to generate sourcecode
        bytecode_t *instr = &kernel.program[j];

        //
        // Skip code generation for system and extensions
        if ((instr->op == SYSTEM) || (instr->op == EXTENSION)) {
            continue;
        }

        //
        // The operation (ewise, reduction, scan, random, range).
        ctemplate::TemplateDictionary* operation_d = kernel_d.AddIncludeDictionary("OPERATIONS");
        string tf = template_filename(
            kernel,
            j,
            optimized
        );
        operation_d->SetFilename(tf);

        //
        // The operator +, -, /, min, max, sin, sqrt, etc...
        //
        ctemplate::TemplateDictionary* operator_d = operation_d->AddSectionDictionary("OPERATORS");
        operator_d->SetValue("OPERATOR", operator_cexpr(instr, kernel->args[instr->out].type));

        //
        // Reduction and scan specific expansions
        // TODO: fix for multiple instructions
        if ((instr->op == REDUCE) || (instr->op == SCAN)) {
            operation_d->SetValue("TYPE_OUTPUT", enum_to_ctypestr(instr->operand[0].base->type));
            operation_d->SetValue("TYPE_INPUT",  enum_to_ctypestr(instr->operand[1].base->type));
            operation_d->SetValue("TYPE_AXIS",  "int64_t");
            if (instr->oper == ADD) {
                operation_d->SetValue("NEUTRAL_ELEMENT", std::to_string(0));
            } else if (instr->oper == MULTIPLY) {
                operation_d->SetValue("NEUTRAL_ELEMENT", std::to_string(1));
            }
        }
        operation_d->SetValue("NR_OUTPUT", std::to_string(instr->out));
        operation_d->SetValue("NR_FINPUT", std::to_string(instr->in1));  // Not all have
        operation_d->SetValue("NR_SINPUT", std::to_string(instr->in2));  // Not all have

        //
        // Fill out the instruction operands globally such that they
        // are available to both for the kernel argument unpacking, the operations and the operators.
        //
        // TODO: this should actually distinguish between the total set of operands
        // and those used for a single instruction depending on the amount of loops that can be
        // fused
        //
        int nops_instr = bh_operands(instr->opcode);
        for(int i=0; i<nops_instr; ++i, ++nops_kernel) {        // Operand dict
            ctemplate::TemplateDictionary* argument_d = kernel_d.AddSectionDictionary("ARGUMENT");
            ctemplate::TemplateDictionary* operand_d  = operation_d->AddSectionDictionary("OPERAND");

            argument_d->SetIntValue("NR", nops_kernel);
            operand_d->SetIntValue("NR",  nops_kernel);
            if (bh_is_constant(&instr->operand[i])) {   // Constant
                argument_d->SetValue(                   // As argument
                    "TYPE",
                    enum_to_ctypestr(instr->constant.type)
                );
                operand_d->SetValue(                    // As operand
                    "TYPE",
                    enum_to_ctypestr(instr->constant.type)
                );  
            } else {                                    // Array
                argument_d->SetValue(                   // As argument
                    "TYPE", 
                    enum_to_ctypestr(instr->operand[i].base->type)
                );
                argument_d->ShowSection("ARRAY");
                operand_d->SetValue(                    // As operand
                    "TYPE", 
                    enum_to_ctypestr(instr->operand[i].base->type)
                );
                operand_d->ShowSection("ARRAY");
            }
        }
    }

    //
    // Fill out the template and return the generated sourcecode
    //
    ctemplate::ExpandTemplate(
        "kernel.tpl", 
        strip_mode,
        &kernel_d,
        &sourcecode
    );

    return sourcecode;
}

/**
 *  Create a symbol for the kernel.
 *
 *  NOTE: System opcodes are ignored.
 *        If a kernel consists of nothing but system opcodes
 *        then no symbol will be created.
 */
bool symbolize(bh_kernel_t &kernel, bh_intp const optimized) {

    std::string symbol_opcode, 
                symbol_lmask,
                symbol_tsig,
                symbol_ndim;

    kernel.symbol   = "";
    kernel.ninstr_nonsys  = 0;        // Count the amount of system opcodes.
    for (int i=0; i<kernel.ninstr; ++i) {

        bh_instruction *instr = kernel.instr[i];
    
        // Do not include system opcodes in the kernel symbol.
        if ((instr->opcode >= BH_DISCARD) && (instr->opcode <= BH_NONE)) {  
            continue;
        }
        kernel.ninstr_nonsys++;

        int tsig    = bh_type_sig(instr);
        int lmask   = bh_layoutmask(instr);

        int ndim;
        switch (instr->opcode) {   // [OPCODE_SWITCH]
            case BH_ADD_REDUCE:
            case BH_MULTIPLY_REDUCE:
            case BH_MINIMUM_REDUCE:
            case BH_MAXIMUM_REDUCE:
            case BH_LOGICAL_AND_REDUCE:
            case BH_BITWISE_AND_REDUCE:
            case BH_LOGICAL_OR_REDUCE:
            case BH_LOGICAL_XOR_REDUCE:
            case BH_BITWISE_OR_REDUCE:
            case BH_BITWISE_XOR_REDUCE:
                ndim = instr->operand[1].ndim;
                break;

            default:
                ndim = instr->operand[0].ndim;
                break;
        }

        symbol_opcode  += std::string(bh_opcode_to_cstr_short(instr->opcode));
        symbol_tsig    += std::string(bh_typesig_to_shorthand(tsig));
        symbol_lmask   += std::string(bh_layoutmask_to_shorthand(lmask));

        if (optimized && (ndim <= 3)) {        // Optimized
            symbol_ndim += std::to_string(ndim);
        } else {
            symbol_ndim += std::string("N");
        }
        symbol_ndim += "D";

        kernel.tsig[i]  = tsig;
        kernel.lmask[i] = lmask;
    }

    //
    //  If the kernel contained nothing but system opcodes, then
    //  a symbol must not be created.
    //
    if (kernel.ninstr_nonsys>0) {
        kernel.symbol = "BH_" + \
                        symbol_opcode  + "_" +\
                        symbol_tsig    + "_" +\
                        symbol_lmask   + "_" +\
                        symbol_ndim;    
    }
    return true;
}
