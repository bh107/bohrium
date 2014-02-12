/**
 *  Choose the template.
 *
 *  Contract: Do not call this for system or extension operations.
 */
string template_filename(bh_kernel_t& kernel, int pc, bh_intp optimized)
{
    string tpl_ndim   = "nd.",
           tpl_opcode,
           tpl_layout = "strided.";

    tac_t* tac = &kernel.program[pc];
    int ndim = (tac->op == REDUCE)         ? \
               kernel.scope[tac->in1].ndim : \
               kernel.scope[tac->out].ndim;
    int lmask = kernel.lmask[pc];

    switch (tac->op) {                    // OPCODE_SWITCH
        case MAP:

            tpl_opcode  = "ewise.";
            if ((optimized) && ((lmask == LMASK_CC) || \
                                (lmask == LMASK_CK))) {
                tpl_layout  = "cont.";
            } else if ((optimized) && (ndim == 1)) {
                tpl_ndim = "1d.";
            } else if ((optimized) && (ndim == 2)) {
                tpl_ndim = "2d.";
            } else if ((optimized) && (ndim == 3)) {
                tpl_ndim = "3d.";
            }
            break;

        case ZIP:
            tpl_opcode  = "ewise.";
            if ((optimized) && (
                (lmask == LMASK_CCC) || (lmask == LMASK_CKC) || (lmask == LMASK_CCK)
                )) {
                tpl_layout  = "cont.";
            } else if ((optimized) && (ndim == 1)) {
                tpl_ndim = "1d.";
            } else if ((optimized) && (ndim == 2)) {
                tpl_ndim = "2d.";
            } else if ((optimized) && (ndim == 3)) {
                tpl_ndim = "3d.";
            }
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

        case GENERATE:
            switch(tac->oper) {
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

        default:
            printf("template_filename: Err=[Unsupported operation %d.]\n", tac->oper);
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
string specialize(bh_kernel_t& kernel, bh_intp const optimized) {

    string sourcecode  = "";

    ctemplate::TemplateDictionary kernel_d("KERNEL");   // Kernel - function wrapping code
    kernel_d.SetValue("SYMBOL", kernel.symbol);

    for(int j=0; j<kernel.ninstr; ++j) {
        
        //
        // Grab the tacuction for which to generate sourcecode
        tac_t *tac = &kernel.program[j];

        //
        // Skip code generation for system and extensions
        if ((tac->op == SYSTEM) || (tac->op == EXTENSION)) {
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
        operator_d->SetValue("OPERATOR", operator_cexpr(tac->op, tac->oper, kernel.scope[tac->out].type));

        //
        // Reduction and scan specific expansions
        // TODO: fix for multiple tacuctions
        if ((tac->op == REDUCE) || (tac->op == SCAN)) {
            operation_d->SetValue("TYPE_OUTPUT", enum_to_ctypestr(kernel.scope[tac->out].type));
            operation_d->SetValue("TYPE_INPUT",  enum_to_ctypestr(kernel.scope[tac->in1].type));
            operation_d->SetValue("TYPE_AXIS",  "int64_t");
            if (tac->oper == ADD) {
                operation_d->SetValue("NEUTRAL_ELEMENT", std::to_string(0));
            } else if (tac->oper == MULTIPLY) {
                operation_d->SetValue("NEUTRAL_ELEMENT", std::to_string(1));
            }
        }
        operation_d->SetValue("NR_OUTPUT", std::to_string(tac->out));
        operation_d->SetValue("NR_FINPUT", std::to_string(tac->in1));  // Not all have
        operation_d->SetValue("NR_SINPUT", std::to_string(tac->in2));  // Not all have

        //
        // Fill out the tacuction operands globally such that they
        // are available to both for the kernel argument unpacking, the operations and the operators.
        //
        // TODO: this should actually distinguish between the total set of operands
        // and those used for a single tacuction depending on the amount of loops that can be
        // fused
        //
        int nops_tac = noperands(tac);
        for(int i=0; i<nops_tac; ++i) {        // Operand dict
            ctemplate::TemplateDictionary* argument_d = kernel_d.AddSectionDictionary("ARGUMENT");
            ctemplate::TemplateDictionary* operand_d  = operation_d->AddSectionDictionary("OPERAND");

            argument_d->SetIntValue("NR", tac->out);
            operand_d->SetIntValue("NR",  tac->out);
            argument_d->SetValue("TYPE", enum_to_ctypestr(kernel.scope[tac->out]);
            if (tac->scope[out].layout != CONSTANT) {
                argument_d->ShowSection("ARRAY");
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
 *  NOTE: System and extension opcodes are ignored.
 *        If a kernel consists of nothing but system and/or extension
 *        opcodes then the symbol will be the empty string "".
 */
bool symbolize(bh_kernel_t &kernel, bh_intp const optimized) {

    std::string symbol_opcode, 
                symbol_lmask,
                symbol_tsig,
                symbol_ndim;

    kernel.symbol   = "";

    for (int i=0; i<kernel.ninstr; ++i) {
        tac_t *instr = kernel.program[i];

        // Do not include system opcodes in the kernel symbol.
        if ((instr->opcode == SYSTEM) || (instr->opcode == EXTENSION)) {
            continue;
        }
        
        symbol_opcode  += std::string(bh_opcode_to_cstr_short(instr->opcode));
        symbol_tsig    += std::string(bh_typesig_to_shorthand(kernel->tsig[i]));
        symbol_lmask   += std::string(bh_layoutmask_to_shorthand(kernel->lmask[i]));
    
        int ndim = kernel->args[instr->out];
        if (instr->op == REDUCE) {
            ndim = kernel->args[instr->in1];
        }
        if (optimized && (ndim <= 3)) {        // Optimized
            symbol_ndim += std::to_string(ndim);
        } else {
            symbol_ndim += std::string("N");
        }
        symbol_ndim += "D";

        kernel.tsig[i]  = tsig;
        kernel.lmask[i] = lmask;
    }

    if (kernel.ninstr_nonsys>0) {
        kernel.symbol = "BH_" + \
                        symbol_opcode  + "_" +\
                        symbol_tsig    + "_" +\
                        symbol_lmask   + "_" +\
                        symbol_ndim;    
    }
    return true;
}
