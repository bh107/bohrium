#ifndef __BH_VE_CPU_SPECIALIZER
#define __BH_VE_CPU_SPECIALIZER

#include <ctemplate/template.h>

void specializer_init()
{
    ctemplate::mutable_default_template_cache()->SetTemplateRootDirectory(template_path);
    ctemplate::LoadTemplate("license.tpl",  ctemplate::STRIP_BLANK_LINES);
    ctemplate::LoadTemplate("skeleton.tpl", ctemplate::STRIP_BLANK_LINES);

    ctemplate::LoadTemplate("range.1d.tpl",    ctemplate::STRIP_BLANK_LINES);
    ctemplate::LoadTemplate("random.1d.tpl",   ctemplate::STRIP_BLANK_LINES);

    ctemplate::LoadTemplate("ewise.1d.tpl",      ctemplate::STRIP_BLANK_LINES);
    ctemplate::LoadTemplate("ewise.2d.tpl",      ctemplate::STRIP_BLANK_LINES);
    ctemplate::LoadTemplate("ewise.3d.tpl",      ctemplate::STRIP_BLANK_LINES);
    ctemplate::LoadTemplate("ewise.nd.ccc.tpl",  ctemplate::STRIP_BLANK_LINES);
    ctemplate::LoadTemplate("ewise.nd.tpl",      ctemplate::STRIP_BLANK_LINES);

    ctemplate::LoadTemplate("reduce.1d.tpl", ctemplate::STRIP_BLANK_LINES);
    ctemplate::LoadTemplate("reduce.2d.tpl", ctemplate::STRIP_BLANK_LINES);
    ctemplate::LoadTemplate("reduce.3d.tpl", ctemplate::STRIP_BLANK_LINES);
    ctemplate::LoadTemplate("reduce.nd.tpl", ctemplate::STRIP_BLANK_LINES);

    ctemplate::LoadTemplate("scan.1d.tpl",  ctemplate::STRIP_BLANK_LINES);
    ctemplate::LoadTemplate("scan.2d.tpl",  ctemplate::STRIP_BLANK_LINES);
    ctemplate::LoadTemplate("scan.3d.tpl",  ctemplate::STRIP_BLANK_LINES);
    ctemplate::LoadTemplate("scan.nd.tpl",  ctemplate::STRIP_BLANK_LINES);

    ctemplate::mutable_default_template_cache()->Freeze();
}

/**
 *  Choose the template.
 */
string template_filename(bh_instruction *instr, bh_intp optimized, bh_intp ndim, int lmask)
{
    string tpl_ndim,
           tpl_opcode,
           tpl_lmask = "";

    if (optimized && (ndim <= 3)) {
        tpl_ndim = to_string(ndim) + "d.";
    } else {
        tpl_ndim = "nd.";
    }

    switch (instr->opcode) {                    // OPCODE_SWITCH

        case BH_RANDOM:

            tpl_opcode = "random.";
            break;

        case BH_RANGE:

            tpl_opcode = "range.";
            break;

        case BH_ADD_ACCUMULATE:
        case BH_MULTIPLY_ACCUMULATE:

            tpl_opcode = "scan.";
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
            if ((lmask == (A0_CONTIGUOUS + A1_CONTIGUOUS    + A2_CONTIGUOUS)) || \
                (lmask == (A0_CONTIGUOUS + A1_CONSTANT      + A2_CONTIGUOUS)) || \
                (lmask == (A0_CONTIGUOUS + A1_CONTIGUOUS    + A2_CONSTANT))) {
                tpl_ndim    = "nd.";
                tpl_lmask   = "ccc.";
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
            if ((lmask == (A0_CONTIGUOUS + A1_CONTIGUOUS)) || \
                (lmask == (A0_CONTIGUOUS + A1_CONSTANT))) {
                tpl_ndim  = "nd.";
                tpl_lmask = "ccc.";
            }
            break;

        default:
            printf("specializer: Err=[Unsupported opcode.] {\n");
            bh_pprint_instr(instr);
            printf("}\n");
            throw runtime_error("cpu-ve: Failed specializing code.");
    }

    return tpl_opcode + tpl_ndim + tpl_lmask + "tpl";
}

/**
 *  Create a symbol for the kernel.
 *
 *  NOTE: System opcodes are ignored.
 */
bool symbolize(bh_kernel_t &kernel, bh_intp const optimized) {

    int non_system = 0;
    std::string symbol_opcode, 
                symbol_lmask,
                symbol_tsig,
                symbol_ndim;

    for (int i=0; i<kernel.ninstr; ++i) {

        bh_instruction *instr = kernel.instr[i];
    
        // Do not include system opcodes in the kernel symbol.
        if ((instr->opcode >= BH_DISCARD) && (instr->opcode <= BH_NONE)) {  
            continue;
        }
        ++non_system;

        int tsig    = bh_typesig(instr);
        int lmask   = bh_layoutmask(instr);
        int ndim;

        if (!bh_typesig_check(tsig)) {
            printf("cpu( Invalid type signature[%lld] ): Bridge check yourself! Instruction:\n", (long long)tsig);
            bh_pprint_instr(instr);
            printf("\n");
            return false;
        }
        
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
        kernel.ndim[i]  = ndim;
    }

    //
    //  If the kernel contained nothing but system opcodes, then
    //  a symbol must not be created.
    //
    if (non_system>0) {
        kernel.symbol += "BH_" + \
                        symbol_opcode  + "_" +\
                        symbol_tsig    + "_" +\
                        symbol_lmask   + "_" +\
                        symbol_ndim;    
    }
    return true;
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

    ctemplate::TemplateDictionary skeleton_dict("SKELETON");    // Skeleton code
    skeleton_dict.SetValue("SYMBOL", kernel.symbol);
    ctemplate::ExpandTemplate(
        "skeleton.tpl", 
        ctemplate::STRIP_BLANK_LINES,
        &skeleton_dict,
        &sourcecode
    );

    ctemplate::TemplateDictionary dict("KERNEL");               // Kernel code

    int nops_kernel = 0;
    for(int j=0; j<kernel.ninstr; ++j) {
        bh_instruction *instr = kernel.instr[j];

        //
        // Ignore system opcodes
        //
        if ((instr->opcode >= BH_DISCARD) && (instr->opcode <= BH_NONE)) {  
            continue;
        }

        bh_type type = instr->operand[0].base->type;

        int nops_instr = bh_operands(instr->opcode);

        ctemplate::TemplateDictionary* operator_dict = dict.AddSectionDictionary("LOOP_BODY");
        operator_dict->SetValue("OPERATOR", bhopcode_to_cexpr(instr->opcode, type));

        for(int i=0; i<nops_instr; ++i, ++nops_kernel) {        // Operand dict
            ctemplate::TemplateDictionary* op_dict = dict.AddSectionDictionary("OPERAND");

            op_dict->SetIntValue("NR", nops_kernel);
            if (bh_is_constant(&instr->operand[i])) {           // Constant
                op_dict->SetValue(
                    "TYPE",
                    enum_to_ctypestr(instr->constant.type)
                );  
            } else {                                            // Array
                op_dict->SetValue(
                    "TYPE", 
                    enum_to_ctypestr(instr->operand[i].base->type)
                );
                op_dict->ShowSection("ARRAY");
            }
        }

        //
        // Reduction and scan specific expansions
        //
        if (((instr->opcode >= BH_ADD_REDUCE) && (instr->opcode <= BH_BITWISE_XOR_REDUCE)) || \
            ((instr->opcode >= BH_ADD_ACCUMULATE) && (instr->opcode <= BH_MULTIPLY_ACCUMULATE))) {
            dict.SetValue("TYPE_INPUT", enum_to_ctypestr(instr->operand[1].base->type));
            dict.SetValue("TYPE_AXIS",  "int64_t");
        }
        string tf = template_filename(instr, optimized, kernel.ndim[j], kernel.lmask[j]);
        ctemplate::ExpandTemplate(
            tf,
            ctemplate::STRIP_BLANK_LINES,
            &dict,
            &sourcecode
        );
    }

    return sourcecode;
}

#endif

