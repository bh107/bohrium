#ifndef __BH_VE_CPU_SPECIALIZER
#define __BH_VE_CPU_SPECIALIZER

#include <ctemplate/template.h>

void specializer_init()
{
    ctemplate::mutable_default_template_cache()->SetTemplateRootDirectory(template_path);
    ctemplate::LoadTemplate("license.tpl",  ctemplate::STRIP_BLANK_LINES);
    ctemplate::LoadTemplate("skeleton.tpl", ctemplate::STRIP_BLANK_LINES);

    ctemplate::LoadTemplate("range.tpl",    ctemplate::STRIP_BLANK_LINES);
    ctemplate::LoadTemplate("random.tpl",   ctemplate::STRIP_BLANK_LINES);

    ctemplate::LoadTemplate("ewise.1d.tpl",      ctemplate::STRIP_BLANK_LINES);
    ctemplate::LoadTemplate("ewise.2d.tpl",      ctemplate::STRIP_BLANK_LINES);
    ctemplate::LoadTemplate("ewise.3d.tpl",      ctemplate::STRIP_BLANK_LINES);
    ctemplate::LoadTemplate("ewise.nd.ddd.tpl",  ctemplate::STRIP_BLANK_LINES);
    ctemplate::LoadTemplate("ewise.nd.tpl",      ctemplate::STRIP_BLANK_LINES);

    ctemplate::LoadTemplate("reduce.1d.tpl", ctemplate::STRIP_BLANK_LINES);
    ctemplate::LoadTemplate("reduce.2d.tpl", ctemplate::STRIP_BLANK_LINES);
    ctemplate::LoadTemplate("reduce.3d.tpl", ctemplate::STRIP_BLANK_LINES);
    ctemplate::LoadTemplate("reduce.nd.tpl", ctemplate::STRIP_BLANK_LINES);

    ctemplate::LoadTemplate("scan.1d.tpl",  ctemplate::STRIP_BLANK_LINES);
    ctemplate::LoadTemplate("scan.nd.tpl",  ctemplate::STRIP_BLANK_LINES);

    ctemplate::mutable_default_template_cache()->Freeze();
}

bool symbolize(bh_instruction *instr, bh_sij_t &sij, bh_intp optimized) {

    char symbol_c[500]; // String representation buffers

    sij.instr = instr;
    sij.lmask = bh_layoutmask(sij.instr);       // Layout mask
    sij.tsig  = bh_typesig(sij.instr);          // Type signature
    
    switch (sij.instr->opcode) {    // [OPCODE_SWITCH]

        case BH_NONE:                                   // System opcodes
        case BH_DISCARD:
        case BH_SYNC:
        case BH_FREE:               // Return without a symbol
            return true;
            break;

        case BH_ADD_REDUCE:                             // Reductions
        case BH_MULTIPLY_REDUCE:
        case BH_MINIMUM_REDUCE:
        case BH_MAXIMUM_REDUCE:
        case BH_LOGICAL_AND_REDUCE:
        case BH_BITWISE_AND_REDUCE:
        case BH_LOGICAL_OR_REDUCE:
        case BH_LOGICAL_XOR_REDUCE:
        case BH_BITWISE_OR_REDUCE:
        case BH_BITWISE_XOR_REDUCE:
            sij.ndim = sij.instr->operand[1].ndim;     // Dimensions
            break;

        default:                                        // Built-in
            sij.ndim = sij.instr->operand[0].ndim;     // Dimensions
            break;
    }

    // String representation
    if (optimized && (sij.ndim <= 3)) {        // Optimized                       
        sprintf(symbol_c, "BH_%s_%s_%s_%lldD",
            bh_opcode_to_cstr_short(sij.instr->opcode),
            bh_typesig_to_shorthand(sij.tsig),
            bh_layoutmask_to_shorthand(sij.lmask),
            (long long)sij.ndim
        );
    } else {                                    // General-case
        sprintf(symbol_c, "BH_%s_%s_%s_ND",
            bh_opcode_to_cstr_short(sij.instr->opcode),
            bh_typesig_to_shorthand(sij.tsig),
            bh_layoutmask_to_shorthand(sij.lmask)
        );
    }

    if (!bh_typesig_check(sij.tsig)) {
        printf("cpu( Invalid type signature[%lld] ): Bridge check yourself! Instruction:\n", (long long)sij.tsig);
        bh_pprint_instr(instr);
        printf("\n");
        return false;
    } else {
        sij.symbol = string(symbol_c);      // Assign the symbol
        return true;
    }
}

string specialize(bh_sij_t &sij, bh_intp optimized) {

    char template_fn[500];   // NOTE: constants like these are often traumatizing!

    bool cres = false;

    ctemplate::TemplateDictionary skeleton_dict("SKELETON");
    ctemplate::TemplateDictionary dict("codegen");

    bh_type type = sij.instr->operand[0].base->type;// Magic parameter to cexpr-function

    skeleton_dict.SetValue("SYMBOL", sij.symbol);
    dict.ShowSection("LOOP_BODY");  // We only have a single expression so we just show it.
    dict.SetValue("OPERATOR",   bhopcode_to_cexpr(sij.instr->opcode, type));

    int nops = bh_operands(sij.instr->opcode);

    for(int i=0; i<nops; ++i) {     // Operand dict
        ctemplate::TemplateDictionary* op_dict   = dict.AddSectionDictionary("OPERAND");

        op_dict->SetIntValue("NR", i);
        if (bh_is_constant(&sij.instr->operand[i])) {    // Constant
            op_dict->SetValue(
                "TYPE",
                enum_to_ctypestr(sij.instr->constant.type)
            );  
        } else {                        // Array
            op_dict->SetValue(
                "TYPE", 
                enum_to_ctypestr(sij.instr->operand[i].base->type)
            );
            op_dict->ShowSection("ARRAY");
        }
    }

    switch (sij.instr->opcode) {                    // OPCODE_SWITCH

        case BH_RANDOM:
            sprintf(template_fn, "random.tpl");
            cres = true;
            break;

        case BH_RANGE:
            sprintf(template_fn, "range.tpl");
            cres = true;
            break;

        case BH_ADD_ACCUMULATE:
        case BH_MULTIPLY_ACCUMULATE:

            dict.SetValue("TYPE_INPUT", enum_to_ctypestr(sij.instr->operand[1].base->type));
            dict.SetValue("TYPE_AXIS",  "int64_t");
            sprintf(template_fn, "scan.1d.tpl");

            cres = true;
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

            dict.SetValue("TYPE_INPUT", enum_to_ctypestr(sij.instr->operand[1].base->type));
            dict.SetValue("TYPE_AXIS", "int64_t");
            if (optimized && (sij.ndim <= 3)) {
                sprintf(template_fn, "reduce.%lldd.tpl", (long long)sij.ndim);
            } else {
                sprintf(template_fn, "reduce.nd.tpl");
            }

            cres = true;
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

            if ((sij.lmask == (A0_CONTIGUOUS + A1_CONTIGUOUS    + A2_CONTIGUOUS)) || \
                (sij.lmask == (A0_CONTIGUOUS + A1_CONSTANT      + A2_CONTIGUOUS)) || \
                (sij.lmask == (A0_CONTIGUOUS + A1_CONTIGUOUS    + A2_CONSTANT))) {
                sprintf(template_fn, "ewise.nd.ddd.tpl");
            } else {
                if (optimized && (sij.ndim<=3)) {
                    sprintf(template_fn, "ewise.%lldd.tpl", (long long)sij.ndim);
                } else {
                    sprintf(template_fn, "ewise.nd.tpl");
                }
            }

            cres = true;
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

            if ((sij.lmask == (A0_CONTIGUOUS + A1_CONTIGUOUS)) || \
                (sij.lmask == (A0_CONTIGUOUS + A1_CONSTANT))) {
                sprintf(template_fn, "ewise.nd.ddd.tpl");
            } else {
                if (optimized && (sij.ndim<=3)) {
                    sprintf(template_fn, "ewise.%lldd.tpl", (long long)sij.ndim);
                } else {
                    sprintf(template_fn, "ewise.nd.tpl");
                }
            }

            cres = true;
            break;

        default:
            printf("specializer: Err=[Unsupported opcode.] {\n");
            bh_pprint_instr(sij.instr);
            printf("}\n");
    }

    if (!cres) {
        throw runtime_error("cpu-ve: Failed specializing code.");
    }

    string sourcecode  = "";
    ctemplate::ExpandTemplate(
        "skeleton.tpl", 
        ctemplate::STRIP_BLANK_LINES,
        &skeleton_dict,
        &sourcecode
    );

    ctemplate::ExpandTemplate(
        template_fn,
        ctemplate::STRIP_BLANK_LINES,
        &dict,
        &sourcecode
    );

    return sourcecode;
}

bool symbolize(bh_kernel_t &kernel, bh_intp const optimized) {

    std::string symbol_opcode, 
                symbol_lmask,
                symbol_tsig,
                symbol_ndim;

    for (int i=0; i<kernel.ninstr; ++i) {

        bh_instruction *instr = &kernel.instr[i];
        int tsig    = bh_typesig(instr);
        int lmask   = bh_layoutmask(instr);
        int ndim;

        if (!bh_typesig_check(tsig)) {
            printf("cpu( Invalid type signature[%lld] ): Bridge check yourself! Instruction:\n", (long long)tsig);
            printf("\n");
            return false;
        }
        
        switch (instr->opcode) {   // [OPCODE_SWITCH]
            case BH_NONE:
                ndim = 0;
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

    kernel.symbol += symbol_opcode  + "_" +\
                     symbol_tsig    + "_" +\
                     symbol_lmask   + "_" +\
                     symbol_ndim;    // Assign the symbol

    return true;
}

string specialize(bh_kernel_t &kernel, bh_intp const optimized) {

    std::string template_fn;

    bool cres = false;

    ctemplate::TemplateDictionary skeleton_dict("SKELETON");
    ctemplate::TemplateDictionary dict("codegen");

    dict.ShowSection("LOOP_BODY");  // We only have a single expression so we just show it.

    for(int j=0; j<kernel.ninstr; ++j) {
        bh_instruction *instr = &kernel.instr[j];
        int lmask = kernel.lmask[j];
        int ndim  = kernel.ndim[j];
        int nops = bh_operands(instr->opcode);

        ctemplate::TemplateDictionary* operator_dict = dict.AddSectionDictionary("LOOP_BODY");
        bh_type type = instr->operand[0].base->type;    // Magic parameter to cexpr-function
        operator_dict->SetValue("OPERATOR", bhopcode_to_cexpr(instr->opcode, type));

        for(int i=0; i<nops; ++i) {     // Operand dict
            ctemplate::TemplateDictionary* op_dict = dict.AddSectionDictionary("OPERAND");

            op_dict->SetIntValue("NR", i);
            if (bh_is_constant(&instr->operand[i])) {    // Constant
                op_dict->SetValue(
                    "TYPE",
                    enum_to_ctypestr(instr->constant.type)
                );  
            } else {                        // Array
                op_dict->SetValue(
                    "TYPE", 
                    enum_to_ctypestr(instr->operand[i].base->type)
                );
                op_dict->ShowSection("ARRAY");
            }
        }

        switch (instr->opcode) {                    // OPCODE_SWITCH

            case BH_RANDOM:
                template_fn = "random.tpl";
                cres = true;
                break;

            case BH_RANGE:
                template_fn = "range.tpl";
                cres = true;
                break;

            case BH_ADD_ACCUMULATE:
            case BH_MULTIPLY_ACCUMULATE:

                dict.SetValue("TYPE_INPUT", enum_to_ctypestr(instr->operand[1].base->type));
                dict.SetValue("TYPE_AXIS",  "int64_t");
                template_fn = "scan.1d.tpl";

                cres = true;
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

                dict.SetValue("TYPE_INPUT", enum_to_ctypestr(instr->operand[1].base->type));
                dict.SetValue("TYPE_AXIS", "int64_t");
                if (optimized && (ndim <= 3)) {
                    template_fn = "reduce.";
                    template_fn += std::to_string(ndim);
                    template_fn += "d.tpl";
                } else {
                    template_fn = "reduce.nd.tpl";
                }

                cres = true;
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

                if ((lmask == (A0_CONTIGUOUS + A1_CONTIGUOUS    + A2_CONTIGUOUS)) || \
                    (lmask == (A0_CONTIGUOUS + A1_CONSTANT      + A2_CONTIGUOUS)) || \
                    (lmask == (A0_CONTIGUOUS + A1_CONTIGUOUS    + A2_CONSTANT))) {
                    template_fn = "ewise.nd.ddd.tpl";
                } else {
                    if (optimized && (ndim<=3)) {
                        template_fn = "ewise.";
                        template_fn = std::to_string(ndim);
                        template_fn = "d.tpl";
                    } else {
                        template_fn = "ewise.nd.tpl";
                    }
                }

                cres = true;
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

                if ((lmask == (A0_CONTIGUOUS + A1_CONTIGUOUS)) || \
                    (lmask == (A0_CONTIGUOUS + A1_CONSTANT))) {
                    template_fn = "ewise.nd.ddd.tpl";
                } else {
                    if (optimized && (ndim<=3)) {
                        template_fn = "ewise.";
                        template_fn = std::to_string(ndim);
                        template_fn = "d.tpl";
                    } else {
                        template_fn = "ewise.nd.tpl";
                    }
                }

                cres = true;
                break;

            default:
                printf("specializer: Err=[Unsupported opcode.] {\n");
                bh_pprint_instr(instr);
                printf("}\n");
        }
    }

    if (!cres) {
        throw runtime_error("cpu-ve: Failed specializing code.");
    }

    string sourcecode  = "";

    skeleton_dict.SetValue("SYMBOL", kernel.symbol);
    ctemplate::ExpandTemplate(
        "skeleton.tpl", 
        ctemplate::STRIP_BLANK_LINES,
        &skeleton_dict,
        &sourcecode
    );

    ctemplate::ExpandTemplate(
        template_fn,
        ctemplate::STRIP_BLANK_LINES,
        &dict,
        &sourcecode
    );

    return sourcecode;
}

#endif

