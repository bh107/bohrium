#ifndef __BH_VE_DYNAMITE_SPECIALIZER
#define __BH_VE_DYNAMITE_SPECIALIZER

#include <ctemplate/template.h>  

std::string specialize(std::string symbol, bh_instruction *instr) {

    bh_random_type *random_args;

    ctemplate::TemplateDictionary dict("codegen");
    dict.ShowSection("license");
    dict.ShowSection("include");

    bool cres = false;

    char template_fn[250];   // NOTE: constants like these are often traumatizing!
    char dims_str[10];
    int64_t dims = instr->operand[0].ndim;  // Reductions overwrite this

    switch (instr->opcode) {                    // OPCODE_SWITCH

        case BH_USERFUNC:                       // Extensions
            if (instr->userfunc->id == random_impl_id) {
                dict.SetValue("SYMBOL",     symbol);
                dict.SetValue("TYPE_A0",    bhtype_to_ctype(random_args->operand[0].base->type));
                dict.SetValue("TYPE_A0_SHORTHAND", bhtype_to_shorthand(random_args->operand[0].base->type));
                sprintf(template_fn, "%s/random.tpl", template_path);

                cres = true;
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

            dict.SetValue("OPERATOR", bhopcode_to_cexpr(instr->opcode));
            dict.SetValue("SYMBOL", symbol);
            dict.SetValue("TYPE_A0", bhtype_to_ctype(instr->operand[0].base->type));
            dict.SetValue("TYPE_A1", bhtype_to_ctype(instr->operand[1].base->type));

            dims = instr->operand[1].ndim;
            if (1 == dims) {
                sprintf(template_fn, "%s/reduction.1d.tpl", template_path);
            } else if (2 == dims) {
                sprintf(template_fn, "%s/reduction.2d.tpl", template_path);
            } else {
                sprintf(template_fn, "%s/reduction.nd.tpl", template_path);
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

            dict.SetValue("OPERATOR", bhopcode_to_cexpr(instr->opcode));
            dict.ShowSection("binary");
            if (bh_is_constant(&instr->operand[2])) {
                dict.SetValue("SYMBOL", symbol);
                dict.SetValue("TYPE_A0", bhtype_to_ctype(instr->operand[0].base->type));
                dict.SetValue("TYPE_A1", bhtype_to_ctype(instr->operand[1].base->type));
                dict.SetValue("TYPE_A2", bhtype_to_ctype(instr->constant.type));
                dict.ShowSection("a1_dense");
                dict.ShowSection("a2_scalar");
            } else if (bh_is_constant(&instr->operand[1])) {
                dict.SetValue("SYMBOL", symbol);
                dict.SetValue("TYPE_A0", bhtype_to_ctype(instr->operand[0].base->type));
                dict.SetValue("TYPE_A1", bhtype_to_ctype(instr->constant.type));
                dict.SetValue("TYPE_A2", bhtype_to_ctype(instr->operand[2].base->type));
                dict.ShowSection("a1_scalar");
                dict.ShowSection("a2_dense");
            } else {
                dict.SetValue("SYMBOL", symbol);
                dict.SetValue("TYPE_A0", bhtype_to_ctype(instr->operand[0].base->type));
                dict.SetValue("TYPE_A1", bhtype_to_ctype(instr->operand[1].base->type));
                dict.SetValue("TYPE_A2", bhtype_to_ctype(instr->operand[2].base->type));
                dict.ShowSection("a1_dense");
                dict.ShowSection("a2_dense");
            }
            if (1 == dims) {
                sprintf(template_fn, "%s/traverse.1d.tpl", template_path);
            } else if (2 == dims) {
                sprintf(template_fn, "%s/traverse.2d.tpl", template_path);
            } else if (3 == dims) {
                sprintf(template_fn, "%s/traverse.3d.tpl", template_path);
            } else {
                sprintf(template_fn, "%s/traverse.nd.tpl", template_path);
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

            dict.SetValue("OPERATOR", bhopcode_to_cexpr(instr->opcode));
            dict.ShowSection("unary");
            if (bh_is_constant(&instr->operand[1])) {
                dict.SetValue("SYMBOL", symbol);
                dict.SetValue("TYPE_A0", bhtype_to_ctype(instr->operand[0].base->type));
                dict.SetValue("TYPE_A1", bhtype_to_ctype(instr->constant.type));
                dict.ShowSection("a1_scalar");
            } else {
                dict.SetValue("SYMBOL", symbol);
                dict.SetValue("TYPE_A0", bhtype_to_ctype(instr->operand[0].base->type));
                dict.SetValue("TYPE_A1", bhtype_to_ctype(instr->operand[1].base->type));
                dict.ShowSection("a1_dense");
            } 
            if (1 == dims) {
                sprintf(template_fn, "%s/traverse.1d.tpl", template_path);
            } else if (2 == dims) {
                sprintf(template_fn, "%s/traverse.2d.tpl", template_path);
            } else if (3 == dims) {
                sprintf(template_fn, "%s/traverse.3d.tpl", template_path);
            } else {
                sprintf(template_fn, "%s/traverse.nd.tpl", template_path);
            }
            cres = true;

            break;

        default:                            // Shit hit the fan
            printf("Dynamite: Err=[Unsupported ufunc...\n");
            res = BH_ERROR;
    }

    if (!res) {
        // raise
    }

    std::string sourcecode = "";
    ctemplate::ExpandTemplate(
        template_fn,
        ctemplate::STRIP_BLANK_LINES,
        &dict,
        &sourcecode
    );

    return sourcecode;
}

#endif

