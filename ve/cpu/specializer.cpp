#ifndef __BH_VE_CPU_SPECIALIZER
#define __BH_VE_CPU_SPECIALIZER

#include <ctemplate/template.h>

static ctemplate::Strip strip_mode = ctemplate::STRIP_BLANK_LINES;

void specializer_init()
{
    ctemplate::mutable_default_template_cache()->SetTemplateRootDirectory(template_path);
    ctemplate::LoadTemplate("ewise.cont.nd.tpl", strip_mode);
    ctemplate::LoadTemplate("ewise.strided.1d.tpl", strip_mode);
    ctemplate::LoadTemplate("ewise.strided.2d.tpl", strip_mode);
    ctemplate::LoadTemplate("ewise.strided.3d.tpl", strip_mode);
    ctemplate::LoadTemplate("ewise.strided.nd.tpl", strip_mode);
    ctemplate::LoadTemplate("kernel.tpl", strip_mode);
    ctemplate::LoadTemplate("license.tpl", strip_mode);
    ctemplate::LoadTemplate("random.cont.1d.tpl", strip_mode);
    ctemplate::LoadTemplate("range.cont.1d.tpl", strip_mode);
    ctemplate::LoadTemplate("reduce.strided.1d.tpl", strip_mode);
    ctemplate::LoadTemplate("reduce.strided.2d.tpl", strip_mode);
    ctemplate::LoadTemplate("reduce.strided.3d.tpl", strip_mode);
    ctemplate::LoadTemplate("reduce.strided.nd.tpl", strip_mode);
    ctemplate::LoadTemplate("scan.strided.1d.tpl", strip_mode);
    ctemplate::LoadTemplate("scan.strided.nd.tpl", strip_mode);
    ctemplate::mutable_default_template_cache()->Freeze();
}

/**
 *  Choose the template.
 *
 *  Contract: Do not call this for system or extension operations.
 */
string template_filename(block_t& block, int pc, bh_intp optimized)
{
    string tpl_ndim   = "nd.",
           tpl_opcode,
           tpl_layout = "strided.";

    tac_t* tac = &block.program[pc];
    int ndim = (tac->op == REDUCE)         ? \
               block.scope[tac->in1].ndim : \
               block.scope[tac->out].ndim;
    int lmask = block.lmask[pc];

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
 *  Construct the c-sourcecode for the given block.
 *
 *  NOTE: System opcodes are ignored.
 *
 *  @param optimized The level of optimizations to apply to the generated code.
 *  @param block The block to generate sourcecode for.
 *  @return The generated sourcecode.
 *
 */
string specialize(block_t& block, bh_intp const optimized) {

    string sourcecode  = "";

    ctemplate::TemplateDictionary kernel_d("KERNEL");   // Kernel - function wrapping code
    kernel_d.SetValue("SYMBOL", block.symbol);

    for(int j=0; j<block.ninstr; ++j) {
        
        //
        // Grab the tacuction for which to generate sourcecode
        tac_t *tac = &block.program[j];

        //
        // Skip code generation for system and extensions
        if ((tac->op == SYSTEM) || (tac->op == EXTENSION)) {
            continue;
        }

        //
        // The operation (ewise, reduction, scan, random, range).
        ctemplate::TemplateDictionary* operation_d = kernel_d.AddIncludeDictionary("OPERATIONS");
        string tf = template_filename(
            block,
            j,
            optimized
        );
        operation_d->SetFilename(tf);

        //
        // The operator +, -, /, min, max, sin, sqrt, etc...
        //
        ctemplate::TemplateDictionary* operator_d = operation_d->AddSectionDictionary("OPERATORS");
        operator_d->SetValue("OPERATOR", operator_cexpr(tac->op, tac->oper, block.scope[tac->out].type));

        //
        // Reduction and scan specific expansions
        // TODO: fix for multiple tacuctions
        if ((tac->op == REDUCE) || (tac->op == SCAN)) {
            operation_d->SetValue("TYPE_OUTPUT", enum_to_ctypestr(block.scope[tac->out].type));
            operation_d->SetValue("TYPE_INPUT",  enum_to_ctypestr(block.scope[tac->in1].type));
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

            argument_d->SetValue("TYPE", enum_to_ctypestr(block.scope[tac->out].type));
            argument_d->SetIntValue("NR", tac->out);

            operand_d->SetValue("TYPE", enum_to_ctypestr(block.scope[tac->out].type));
            operand_d->SetIntValue("NR",  tac->out);
            if (block.scope[tac->out].layout != CONSTANT) {
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
 *        If a block consists of nothing but system and/or extension
 *        opcodes then the symbol will be the empty string "".
 */
bool symbolize(block_t &block, bh_intp const optimized) {

    std::string symbol_opcode, 
                symbol_lmask,
                symbol_tsig,
                symbol_ndim;

    block.symbol   = "";

    for (int i=0; i<block.ninstr; ++i) {
        tac_t* tac = &block.program[i];

        // Do not include system opcodes in the kernel symbol.
        if ((tac->op == SYSTEM) || (tac->op == EXTENSION)) {
            continue;
        }
        
        symbol_opcode  += std::string(bh_opcode_to_cstr_short(tac->op));
        symbol_tsig    += std::string(bh_typesig_to_shorthand(block.tsig[i]));
        symbol_lmask   += std::string(bh_layoutmask_to_shorthand(block.lmask[i]));
    
        int ndim = block.scope[tac->out].ndim;
        if (tac->op == REDUCE) {
            ndim = block.scope[tac->in1].ndim;
        }
        if (optimized && (ndim <= 3)) {        // Optimized
            symbol_ndim += std::to_string(ndim);
        } else {
            symbol_ndim += std::string("N");
        }
        symbol_ndim += "D";

        block.tsig[i]  = tsig;
        block.lmask[i] = lmask;
    }

    if (block.omask == (HAS_ARRAY_OP)) {
        block.symbol = "BH_" + \
                        symbol_opcode  + "_" +\
                        symbol_tsig    + "_" +\
                        symbol_lmask   + "_" +\
                        symbol_ndim;    
    }
    return true;
}

#endif
