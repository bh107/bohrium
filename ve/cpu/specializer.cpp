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
 */
string template_filename(bh_instruction *instr, bh_intp optimized, int lmask)
{
    string tpl_ndim = "nd.",
           tpl_opcode,
           tpl_layout = "strided.";

    switch (instr->opcode) {                    // OPCODE_SWITCH

        case BH_RANDOM:
            tpl_opcode = "random.";
            tpl_layout = "cont.";
            tpl_ndim   = "1d.";
            break;

        case BH_RANGE:
            tpl_opcode = "range.";
            tpl_layout = "cont.";
            tpl_ndim   = "1d.";
            break;

        case BH_ADD_ACCUMULATE:
        case BH_MULTIPLY_ACCUMULATE:
            tpl_opcode = "scan.";
            if (optimized && (instr->operand[1].ndim == 1)) {
                tpl_ndim = "1d.";
            }
            break;

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
            tpl_opcode = "reduce.";
            if (optimized && (instr->operand[1].ndim == 1)) {
                tpl_ndim = "1d.";
            } else if (optimized && (instr->operand[1].ndim == 2)) {
                tpl_ndim = "2d.";
            } else if (optimized && (instr->operand[1].ndim == 3)) {
                tpl_ndim = "3d.";
            }
            break;

        case BH_ADD:
        case BH_SUBTRACT:
        case BH_MULTIPLY:
        case BH_DIVIDE:
        case BH_POWER:
        case BH_GREATER:
        case BH_GREATER_EQUAL:
        case BH_LESS:
        case BH_LESS_EQUAL:
        case BH_EQUAL:
        case BH_NOT_EQUAL:
        case BH_LOGICAL_AND:
        case BH_LOGICAL_OR:
        case BH_LOGICAL_XOR:
        case BH_MAXIMUM:
        case BH_MINIMUM:
        case BH_BITWISE_AND:
        case BH_BITWISE_OR:
        case BH_BITWISE_XOR:
        case BH_LEFT_SHIFT:
        case BH_RIGHT_SHIFT:
        case BH_ARCTAN2:
        case BH_MOD:
            
            tpl_opcode  = "ewise.";
            if ((optimized) && ( \
                (lmask == (A0_CONTIGUOUS + A1_CONTIGUOUS + A2_CONTIGUOUS)) || \
                (lmask == (A0_CONTIGUOUS + A1_CONSTANT      + A2_CONTIGUOUS)) || \
                (lmask == (A0_CONTIGUOUS + A1_CONTIGUOUS    + A2_CONSTANT)))) {
                tpl_layout  = "cont.";
            } else if ((optimized) && (instr->operand[0].ndim == 1)) {
                tpl_ndim = "1d.";
            } else if ((optimized) && (instr->operand[0].ndim == 2)) {
                tpl_ndim = "2d.";
            } else if ((optimized) && (instr->operand[0].ndim == 3)) {
                tpl_ndim = "3d.";
            }
            break;

        case BH_IMAG:   // These use the width parameter to switch between
        case BH_REAL:   // different cexpressions
        case BH_ABSOLUTE:
        case BH_LOGICAL_NOT:
        case BH_INVERT:
        case BH_COS:
        case BH_SIN:
        case BH_TAN:
        case BH_COSH:
        case BH_SINH:
        case BH_TANH:
        case BH_ARCSIN:
        case BH_ARCCOS:
        case BH_ARCTAN:
        case BH_ARCSINH:
        case BH_ARCCOSH:
        case BH_ARCTANH:
        case BH_EXP:
        case BH_EXP2:
        case BH_EXPM1:
        case BH_LOG:
        case BH_LOG2:
        case BH_LOG10:
        case BH_LOG1P:
        case BH_SQRT:
        case BH_CEIL:
        case BH_TRUNC:
        case BH_FLOOR:
        case BH_RINT:
        case BH_ISNAN:
        case BH_ISINF:
        case BH_IDENTITY:

            tpl_opcode  = "ewise.";
            if ((optimized) && ((lmask == (A0_CONTIGUOUS + A1_CONTIGUOUS)) || \
                (lmask == (A0_CONTIGUOUS + A1_CONSTANT)))) {
                tpl_layout  = "cont.";
            } else if ((optimized) && (instr->operand[0].ndim == 1)) {
                tpl_ndim = "1d.";
            } else if ((optimized) && (instr->operand[0].ndim == 2)) {
                tpl_ndim = "2d.";
            } else if ((optimized) && (instr->operand[0].ndim == 3)) {
                tpl_ndim = "3d.";
            }
            break;

        default:
            printf("template_filename: Err=[Unsupported opcode.] {\n");
            bh_pprint_instr(instr);
            printf("}\n");
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
string specialize(bh_kernel_t &kernel, bh_intp const optimized) 
{
    string sourcecode  = "";

    ctemplate::TemplateDictionary kernel_d("KERNEL");   // Kernel - function wrapping code
    kernel_d.SetValue("SYMBOL", kernel.symbol);

    int nops_kernel = 0;
    for(int j=0; j<kernel.ninstr; ++j) {
        
        //
        // Grab the instruction for which to generate sourcecode
        bh_instruction *instr = kernel.instr[j];

        //
        // Skip code generation if the instruction has a system opcode
        if ((instr->opcode >= BH_DISCARD) && (instr->opcode <= BH_NONE)) {  
            continue;
        }

        //
        // The operation (ewise, reduction, scan, random, range).
        ctemplate::TemplateDictionary* operation_d = kernel_d.AddIncludeDictionary("OPERATIONS");
        string tf = template_filename(instr, optimized, kernel.lmask[j]);
        operation_d->SetFilename(tf);

        //
        // The operator +, -, /, min, max, sin, sqrt, etc...
        //
        ctemplate::TemplateDictionary* operator_d = operation_d->AddSectionDictionary("OPERATORS");
        bh_type type = instr->operand[0].base->type;
        operator_d->SetValue("OPERATOR", bhopcode_to_cexpr(instr->opcode, type));

        //
        // Reduction and scan specific expansions
        // TODO: fix for multiple instructions
        //
        if (((instr->opcode >= BH_ADD_REDUCE) && (instr->opcode <= BH_BITWISE_XOR_REDUCE)) || \
            ((instr->opcode >= BH_ADD_ACCUMULATE) && (instr->opcode <= BH_MULTIPLY_ACCUMULATE))) {
            operation_d->SetValue("TYPE_OUTPUT", enum_to_ctypestr(instr->operand[0].base->type));
            operation_d->SetValue("TYPE_INPUT", enum_to_ctypestr(instr->operand[1].base->type));
            operation_d->SetValue("TYPE_AXIS",  "int64_t");
        }
        if (instr->opcode == BH_ADD_ACCUMULATE) {
            operation_d->SetIntValue("NEUTRAL_ELEMENT", 0);
        } else if (instr->opcode == BH_MULTIPLY_ACCUMULATE) {
            operation_d->SetIntValue("NEUTRAL_ELEMENT", 1);
        }
        operation_d->SetIntValue("NR_OUTPUT", nops_kernel);
        operation_d->SetIntValue("NR_FINPUT", nops_kernel+1);  // Not all have
        operation_d->SetIntValue("NR_SINPUT", nops_kernel+2);  // Not all have

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
bool symbolize(bh_kernel_t &kernel, bh_intp const optimized)
{
    stringstream symbol_opcode, 
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

        symbol_opcode  << bh_opcode_to_cstr_short(instr->opcode);
        symbol_tsig    << bh_typesig_to_shorthand(tsig);
        symbol_lmask   << bh_layoutmask_to_shorthand(lmask);

        if (optimized && (ndim <= 3)) {        // Optimized
            symbol_ndim << ndim;
        } else {
            symbol_ndim << "N";
        }
        symbol_ndim << "D";

        kernel.tsig[i]  = tsig;
        kernel.lmask[i] = lmask;
    }

    //
    //  If the kernel contained nothing but system opcodes, then
    //  a symbol must not be created.
    //
    if (kernel.ninstr_nonsys>0) {
        kernel.symbol = "BH_" + \
                        symbol_opcode.str()  + "_" +\
                        symbol_tsig.str()    + "_" +\
                        symbol_lmask.str()   + "_" +\
                        symbol_ndim.str();    
    }
    return true;
}


#endif

