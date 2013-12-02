#ifndef __BH_VE_CPU_SPECIALIZER
#define __BH_VE_CPU_SPECIALIZER

#include <ctemplate/template.h>  
void symbolize(bh_instruction *instr, bh_sij_t &sij) {

    char symbol_c[500];             // String representation buffers
    char dims_str[10];

    sij.instr = instr;
    switch (sij.instr->opcode) {                    // [OPCODE_SWITCH]

        case BH_NONE:                           // System opcodes
        case BH_DISCARD:
        case BH_SYNC:
        case BH_FREE:
        case BH_USERFUNC:                       // Extensions
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
            sij.ndims = sij.instr->operand[1].ndim;     // Dimensions
            sij.lmask = bh_layoutmask(sij.instr);       // Layout mask
            sij.tsig  = bh_typesig(sij.instr);          // Type signature

            if (sij.ndims <= 3) {                       // String representation
                sprintf(dims_str, "%ldD", sij.ndims);
            } else {
                sprintf(dims_str, "ND");
            }
            sprintf(symbol_c, "%s_%s_%s_%s",
                bh_opcode_text(sij.instr->opcode),
                dims_str,
                bh_layoutmask_to_shorthand(sij.lmask),
                bh_typesig_to_shorthand(sij.tsig)
            );

            sij.symbol = string(symbol_c);
            break;

        case BH_RANGE:

            sij.ndims = sij.instr->operand[0].ndim;     // Dimensions
            sij.lmask = bh_layoutmask(sij.instr);       // Layout mask
            sij.tsig  = bh_typesig(sij.instr);          // Type signature

            sprintf(symbol_c, "%s_ND_%s_%s",
                bh_opcode_text(sij.instr->opcode),
                bh_layoutmask_to_shorthand(sij.lmask),
                bh_typesig_to_shorthand(sij.tsig)
            );

            sij.symbol = string(symbol_c);
            break;

        default:                                        // Built-in
            
            sij.ndims = sij.instr->operand[0].ndim;     // Dimensions
            sij.lmask = bh_layoutmask(sij.instr);       // Layout mask
            sij.tsig  = bh_typesig(sij.instr);          // Type signature

            if (sij.ndims <= 3) {                       // String representation
                sprintf(dims_str, "%ldD", sij.ndims);
            } else {
                sprintf(dims_str, "ND");
            }
            sprintf(symbol_c, "%s_%s_%s_%s",
                bh_opcode_text(sij.instr->opcode),
                dims_str,
                bh_layoutmask_to_shorthand(sij.lmask),
                bh_typesig_to_shorthand(sij.tsig)
            );

            sij.symbol = string(symbol_c);
            break;
    }
}

string specialize(bh_sij_t &sij) {

    char template_fn[500];   // NOTE: constants like these are often traumatizing!

    bool cres = false;

    ctemplate::TemplateDictionary dict("codegen");
    ctemplate::TemplateDictionary include_dict("INCLUDE");

    switch (sij.instr->opcode) {                    // OPCODE_SWITCH

        case BH_RANGE:
            dict.SetValue("OPERATOR", bhopcode_to_cexpr(sij.instr->opcode));
            dict.SetValue("SYMBOL", sij.symbol);
            dict.SetValue("TYPE_A0", bhtype_to_ctype(sij.instr->operand[0].base->type));
            sprintf(template_fn, "range.tpl");

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

            dict.SetValue("OPERATOR", bhopcode_to_cexpr(sij.instr->opcode));
            dict.SetValue("SYMBOL", sij.symbol);
            dict.SetValue("TYPE_A0", bhtype_to_ctype(sij.instr->operand[0].base->type));
            dict.SetValue("TYPE_A1", bhtype_to_ctype(sij.instr->operand[1].base->type));

            if (sij.ndims <= 3) {
                sprintf(template_fn, "reduction.%ldd.tpl", sij.ndims);
            } else {
                sprintf(template_fn, "reduction.nd.tpl");
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
        case BH_RANDOM:

            dict.SetValue("OPERATOR", bhopcode_to_cexpr(sij.instr->opcode));
            if ((sij.lmask & A2_CONSTANT) == A2_CONSTANT) {
                dict.SetValue("SYMBOL", sij.symbol);
                dict.SetValue("TYPE_A0", bhtype_to_ctype(sij.instr->operand[0].base->type));
                dict.SetValue("TYPE_A1", bhtype_to_ctype(sij.instr->operand[1].base->type));
                dict.SetValue("TYPE_A2", bhtype_to_ctype(sij.instr->constant.type));
                dict.ShowSection("a1_dense");
                dict.ShowSection("a2_scalar");
            } else if ((sij.lmask & A1_CONSTANT) == A1_CONSTANT) {
                dict.SetValue("SYMBOL", sij.symbol);
                dict.SetValue("TYPE_A0", bhtype_to_ctype(sij.instr->operand[0].base->type));
                dict.SetValue("TYPE_A1", bhtype_to_ctype(sij.instr->constant.type));
                dict.SetValue("TYPE_A2", bhtype_to_ctype(sij.instr->operand[2].base->type));
                dict.ShowSection("a1_scalar");
                dict.ShowSection("a2_dense");
            } else {
                dict.SetValue("SYMBOL", sij.symbol);
                dict.SetValue("TYPE_A0", bhtype_to_ctype(sij.instr->operand[0].base->type));
                dict.SetValue("TYPE_A1", bhtype_to_ctype(sij.instr->operand[1].base->type));
                dict.SetValue("TYPE_A2", bhtype_to_ctype(sij.instr->operand[2].base->type));
                dict.ShowSection("a1_dense");
                dict.ShowSection("a2_dense");

            }
            if ((sij.lmask == (A0_DENSE + A1_DENSE    + A2_DENSE)) || \
                (sij.lmask == (A0_DENSE + A1_CONSTANT + A2_DENSE)) || \
                (sij.lmask == (A0_DENSE + A1_DENSE    + A2_CONSTANT))) {
                sprintf(template_fn, "traverse.nd.ddd.tpl");
            } else {
                if (sij.ndims<=3) {
                    sprintf(template_fn, "traverse.%ldd.tpl", sij.ndims);
                } else {
                    sprintf(template_fn, "traverse.nd.tpl");
                }
            }

            cres = true;
            break;

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

            dict.SetValue("OPERATOR", bhopcode_to_cexpr(sij.instr->opcode));
            if ((sij.lmask & A1_CONSTANT) == A1_CONSTANT) {
                dict.SetValue("SYMBOL", sij.symbol);
                dict.SetValue("TYPE_A0", bhtype_to_ctype(sij.instr->operand[0].base->type));
                dict.SetValue("TYPE_A1", bhtype_to_ctype(sij.instr->constant.type));
                dict.ShowSection("a1_scalar");
            } else {
                dict.SetValue("SYMBOL", sij.symbol);
                dict.SetValue("TYPE_A0", bhtype_to_ctype(sij.instr->operand[0].base->type));
                dict.SetValue("TYPE_A1", bhtype_to_ctype(sij.instr->operand[1].base->type));
                dict.ShowSection("a1_dense");
            }

            if ((sij.lmask == (A0_DENSE + A1_DENSE)) || \
                (sij.lmask == (A0_DENSE + A1_CONSTANT))) {
                sprintf(template_fn, "traverse.nd.ddd.tpl");
            } else {

                if (sij.ndims<=3) {
                    sprintf(template_fn, "traverse.%ldd.tpl", sij.ndims);
                } else {
                    sprintf(template_fn, "traverse.nd.tpl");
                }

            }

            cres = true;
            break;

        default:
            printf("cpu-ve: Err=[Unsupported ufunc...]\n");
    }

    if (!cres) {
        throw runtime_error("cpu-ve: Failed specializing code.");
    }

    string sourcecode = "";
    ctemplate::ExpandTemplate("include.tpl", ctemplate::STRIP_BLANK_LINES, &include_dict, &sourcecode);
    ctemplate::ExpandTemplate(
        template_fn,
        ctemplate::STRIP_BLANK_LINES,
        &dict,
        &sourcecode
    );

    return sourcecode;
}

#endif

