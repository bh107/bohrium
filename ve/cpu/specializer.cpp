#include "specializer.hpp"

using namespace std;
namespace bohrium {
namespace engine {
namespace cpu {

Specializer::Specializer(const string template_directory)
: strip_mode(ctemplate::STRIP_BLANK_LINES), template_directory(template_directory)
{
    ctemplate::mutable_default_template_cache()->SetTemplateRootDirectory(template_directory);
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

string Specializer::text()
{
    stringstream ss;
    ss << "Specializer(\"" << template_directory;
    ss << "\", " << strip_mode << ");" << endl;

    return ss.str();
}

/**
 *  Choose the template.
 *
 *  Contract: Do not call this for system or extension operations.
 */
string Specializer::template_filename(Block& block, size_t pc, bool optimized)
{
    string tpl_ndim   = "nd.",
           tpl_opcode,
           tpl_layout = "strided.";

    tac_t& tac = block.program[pc];
    int ndim = (tac.op == REDUCE)         ? \
               block.scope[tac.in1].ndim : \
               block.scope[tac.out].ndim;

    LAYOUT layout_out = block.scope[tac.out].layout, 
           layout_in1 = block.scope[tac.in1].layout,
           layout_in2 = block.scope[tac.in2].layout;

    switch (tac.op) {                    // OPCODE_SWITCH
        case MAP:

            tpl_opcode  = "ewise.";
            if (optimized && \
                ((layout_out == CONTIGUOUS) && \
                 ((layout_in1 == CONTIGUOUS) || (layout_out == CONSTANT))
                )
               ) {
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
            if (optimized && \
               (layout_out == CONTIGUOUS) && \
                (((layout_in1 == CONTIGUOUS) && (layout_in2 == CONTIGUOUS)) || \
                 ((layout_in1 == CONTIGUOUS) && (layout_in2 == CONSTANT)) || \
                 ((layout_in1 == CONSTANT) && (layout_in2 == CONTIGUOUS)) \
                )
               ) {
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
            switch(tac.oper) {
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
            printf("template_filename: Err=[Unsupported operation %d.]\n", tac.oper);
            throw runtime_error("template_filename: No template for opcode.");
    }

    return tpl_opcode + tpl_layout + tpl_ndim  + "tpl";
}

/**
 *  Construct the c-sourcecode for the given tac.
 *  This generates something along the lines of: *a0_current = *a1_current + *a2_current;
 *  For a MAP of the ADD operator.
 */
string tac_operator_cexpr(Block& block, size_t tac_idx)
{
    tac_t& tac  = block.program[tac_idx];
    ETYPE etype = block.scope[tac.out].type;

    switch(tac.oper) {
        /*
        case BH_MAXIMUM_REDUCE:
            return "rvar = rvar < *tmp_current ? *tmp_current : rvar";
        case BH_LOGICAL_AND_REDUCE:
            return "rvar = rvar && *tmp_current";
        case BH_BITWISE_AND_REDUCE:
            return "rvar = rvar | *tmp_current";
        case BH_LOGICAL_OR_REDUCE:
            return "rvar = rvar || *tmp_current";
        case BH_BITWISE_OR_REDUCE:
            return "rvar |= *tmp_current";

        case BH_LOGICAL_XOR_REDUCE:
            return "rvar = !rvar != !*tmp_current";
        case BH_BITWISE_XOR_REDUCE:
            return "rvar = rvar ^ *tmp_current";
        */

        case ADD:
            switch(tac.op) {
                case MAP:
                    return "*a0_current = *a1_current + *a2_current";
                case SCAN:
                    return  "cvar += *a1_current;"
                            "*a0_current = cvar;";
                case REDUCE:
                    return "rvar += *tmp_current";
                default:
                    return "__UNS_OP_FOR_OPER__";
            }
        case SUBTRACT:
            return "*a0_current = *a1_current - *a2_current";
        case MULTIPLY:
            switch(tac.op) {
                case MAP:
                    return "*a0_current = *a1_current * *a2_current";
                case SCAN:
                    return  "cvar *= *a1_current;"
                            "*a0_current = cvar;";
                case REDUCE:
                    return "rvar *= *tmp_current";
                default:
                    return "__UNS_OP_FOR_OPER__";
            }
        case DIVIDE:
            return "*a0_current = *a1_current / *a2_current";
        case POWER:
            switch(etype) {
                case COMPLEX64:
                    return "*a0_current = cpowf( *a1_current, *a2_current )";
                case COMPLEX128:
                    return "*a0_current = cpow( *a1_current, *a2_current )";
                default:
                    return "*a0_current = pow( *a1_current, *a2_current )";
            }
        case GREATER:
            return "*a0_current = *a1_current > *a2_current";
        case GREATER_EQUAL:
            return "*a0_current = *a1_current >= *a2_current";
        case LESS:
            return "*a0_current = *a1_current < *a2_current";
        case LESS_EQUAL:
            return "*a0_current = *a1_current <= *a2_current";
        case EQUAL:
            return "*a0_current = *a1_current == *a2_current";
        case NOT_EQUAL:
            return "*a0_current = *a1_current != *a2_current";
        case LOGICAL_AND:
            return "*a0_current = *a1_current && *a2_current";
        case LOGICAL_OR:
            return "*a0_current = *a1_current || *a2_current";
        case LOGICAL_XOR:
            return "*a0_current = (!*a1_current != !*a2_current)";
        case MAXIMUM:
            return "*a0_current = *a1_current < *a2_current ? *a2_current : *a1_current";
        case MINIMUM:
            switch(tac.op) {
                case MAP:
                    return "*a0_current = *a1_current < *a2_current ? *a1_current : *a2_current";
                case REDUCE:
                    return "rvar = rvar < *tmp_current ? rvar : *tmp_current";                
            }
            return "__ERR_OPER__";

        case BITWISE_AND:
            switch(tac.op) {
                case MAP:
                    return "*a0_current = *a1_current & *a2_current";
                case REDUCE:
                    return "rvar &= *tmp_current";
            }
            return "__ERR_OPER__";

        case BITWISE_OR:
            return "*a0_current = *a1_current | *a2_current";
        case BITWISE_XOR:
            return "*a0_current = *a1_current ^ *a2_current";
        case LEFT_SHIFT:
            return "*a0_current = (*a1_current) << (*a2_current)";
        case RIGHT_SHIFT:
            return "*a0_current = (*a1_current) >> (*a2_current)";
        case ARCTAN2:
            return "*a0_current = atan2( *a1_current, *a2_current )";
        case MOD:
            return "*a0_current = *a1_current - floor(*a1_current / *a2_current) * *a2_current";

        //
        // Unary Elementwise: SQRT, SIN...
        case ABSOLUTE:
            return "*a0_current = *a1_current < 0.0 ? -*a1_current: *a1_current";
        case LOGICAL_NOT:
            return "*a0_current = !*a1_current";
        case INVERT:
            return "*a0_current = ~*a1_current";
        case COS:
            switch(etype) {
                case COMPLEX64:
                    return "*a0_current = ccosf( *a1_current )";
                case COMPLEX128:
                    return "*a0_current = ccos( *a1_current )";
                default:
                    return "*a0_current = cos( *a1_current )";
            }
        case SIN:
            switch(etype) {
                case COMPLEX64:
                    return "*a0_current = csinf( *a1_current )";
                case COMPLEX128:
                    return "*a0_current = csin( *a1_current )";
                default:
                    return "*a0_current = sin( *a1_current )";
            }
        case TAN:
            switch(etype) {
                case COMPLEX64:
                    return "*a0_current = ctanf( *a1_current )";
                case COMPLEX128:
                    return "*a0_current = ctan( *a1_current )";
                default:
                    return "*a0_current = tan( *a1_current )";
            }
        case COSH:
            switch(etype) {
                case COMPLEX64:
                    return "*a0_current = ccoshf( *a1_current )";
                case COMPLEX128:
                    return "*a0_current = ccosh( *a1_current )";
                default:
                    return "*a0_current = cosh( *a1_current )";
            }
        case SINH:
            switch(etype) {
                case COMPLEX64:
                    return "*a0_current = csinhf( *a1_current )";
                case COMPLEX128:
                    return "*a0_current = csinh( *a1_current )";
                default:
                    return "*a0_current = sinh( *a1_current )";
            }
        case TANH:
            switch(etype) {
                case COMPLEX64:
                    return "*a0_current = ctanhf( *a1_current )";
                case COMPLEX128:
                    return "*a0_current = ctanh( *a1_current )";
                default:
                    return "*a0_current = tanh( *a1_current )";
            }
        case ARCSIN:
            switch(etype) {
                case COMPLEX64:
                    return "*a0_current = casinf( *a1_current )";
                case COMPLEX128:
                    return "*a0_current = casin( *a1_current )";
                default:
                    return "*a0_current = asin( *a1_current )";
            }
        case ARCCOS:
            switch(etype) {
                case COMPLEX64:
                    return "*a0_current = cacosf( *a1_current )";
                case COMPLEX128:
                    return "*a0_current = cacos( *a1_current )";
                default:
                    return "*a0_current = acos( *a1_current )";
            }
        case ARCTAN:
            switch(etype) {
                case COMPLEX64:
                    return "*a0_current = catanf( *a1_current )";
                case COMPLEX128:
                    return "*a0_current = catan( *a1_current )";
                default:
                    return "*a0_current = atan( *a1_current )";
            }
        case ARCSINH:
            switch(etype) {
                case COMPLEX64:
                    return "*a0_current = casinhf( *a1_current )";
                case COMPLEX128:
                    return "*a0_current = casinh( *a1_current )";
                default:
                    return "*a0_current = asinh( *a1_current )";
            }
        case ARCCOSH:
            switch(etype) {
                case COMPLEX64:
                    return "*a0_current = cacoshf( *a1_current )";
                case COMPLEX128:
                    return "*a0_current = cacosh( *a1_current )";
                default:
                    return "*a0_current = acosh( *a1_current )";
            }
        case ARCTANH:
            switch(etype) {
                case COMPLEX64:
                    return "*a0_current = catanhf( *a1_current )";
                case COMPLEX128:
                    return "*a0_current = catanh( *a1_current )";
                default:
                    return "*a0_current = atanh( *a1_current )";
            }
        case EXP:
            switch(etype) {
                case COMPLEX64:
                    return "*a0_current = cexpf( *a1_current )";
                case COMPLEX128:
                    return "*a0_current = cexp( *a1_current )";
                default:
                    return "*a0_current = exp( *a1_current )";
            }
        case EXP2:
            switch(etype) {
                case COMPLEX64:
                    return "*a0_current = cpowf( 2, *a1_current )";
                case COMPLEX128:
                    return "*a0_current = cpow( 2, *a1_current )";
                default:
                    return "*a0_current = pow( 2, *a1_current )";
            }
        case EXPM1:
            return "*a0_current = expm1( *a1_current )";
        case LOG:
            switch(etype) {
                case COMPLEX64:
                    return "*a0_current = clogf( *a1_current )";
                case COMPLEX128:
                    return "*a0_current = clog( *a1_current )";
                default:
                    return "*a0_current = log( *a1_current )";
            }
        case LOG2:
            return "*a0_current = log2( *a1_current )";
        case LOG10:
            switch(etype) {
                case COMPLEX64:
                    return "*a0_current = clogf( *a1_current )/log(10)";
                case COMPLEX128:
                    return "*a0_current = clog( *a1_current )/log(10)";
                default:
                    return "*a0_current = log( *a1_current )/log(10)";
            }
        case LOG1P:
            return "*a0_current = log1p( *a1_current )";
        case SQRT:
            switch(etype) {
                case COMPLEX64:
                    return "*a0_current = csqrtf( *a1_current )";
                case COMPLEX128:
                    return "*a0_current = csqrt( *a1_current )";
                default:
                    return "*a0_current = sqrt( *a1_current )";
            }
        case CEIL:
            return "*a0_current = ceil( *a1_current )";
        case TRUNC:
            return "*a0_current = trunc( *a1_current )";
        case FLOOR:
            return "*a0_current = floor( *a1_current )";
            
        case RINT:
            return "*a0_current = (*a1_current > 0.0) ? floor(*a1_current + 0.5) : ceil(*a1_current - 0.5)";
        case ISNAN:
            return "*a0_current = isnan(*a1_current)";
        case ISINF:
            return "*a0_current = isinf(*a1_current)";
        case IDENTITY:
            return "*a0_current = *a1_current";
        case REAL:
            return (etype==FLOAT32) ? "*a0_current = crealf(*a1_current)": "*a0_current = creal(*a1_current)";
        case IMAG:
            return (etype==FLOAT32) ? "*a0_current = cimagf(*a1_current)": "*a0_current = cimagf(*a1_current)";        
    }
    return "__ERR_OPER__";
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
string Specializer::specialize(Block& block, bool optimized)
{
    string sourcecode  = "";

    ctemplate::TemplateDictionary kernel_d("KERNEL");   // Kernel - function wrapping code
    kernel_d.SetValue("SYMBOL", block.symbol);

    for(size_t j=0; j<block.length; ++j) {
        
        //
        // Grab the tac for which to generate sourcecode
        tac_t& tac = block.program[j];

        //
        // Skip code generation for system and extensions
        if ((tac.op == SYSTEM) || (tac.op == EXTENSION)) {
            continue;
        }

        //
        // The operation (ewise, reduction, scan, random, range).
        ctemplate::TemplateDictionary* operation_d  = kernel_d.AddIncludeDictionary("OPERATIONS");
        operation_d->SetFilename(template_filename(block, j, optimized));

        //
        // Reduction and scan specific expansions
        if ((tac.op == REDUCE) || (tac.op == SCAN)) {
            operation_d->SetValue("TYPE_OUTPUT", utils::etype_to_ctype_text(block.scope[tac.out].type));
            operation_d->SetValue("TYPE_INPUT",  utils::etype_to_ctype_text(block.scope[tac.in1].type));
            operation_d->SetValue("TYPE_AXIS",  "int64_t");
            if (tac.oper == ADD) {
                operation_d->SetValue("NEUTRAL_ELEMENT", to_string(0));
            } else if (tac.oper == MULTIPLY) {
                operation_d->SetValue("NEUTRAL_ELEMENT", to_string(1));
            }
        }

        ctemplate::TemplateDictionary* operator_d   = operation_d->AddSectionDictionary("OPERATORS");
        ctemplate::TemplateDictionary* argument_d;  // Block arguments
        ctemplate::TemplateDictionary* operand_d;   // Operator operands

        //
        // The operator +, -, /, min, max, sin, sqrt, etc...
        //        
        operator_d->SetValue("OPERATOR", tac_operator_cexpr(block, j));

        //
        //  The arguments / operands
        switch(utils::tac_noperands(tac)) {
            case 3:
                operation_d->SetValue("NR_SINPUT", to_string(tac.in2));  // Not all have
                operator_d->SetIntValue("NR_SINPUT", tac.out);
                argument_d  = kernel_d.AddSectionDictionary("ARGUMENT");
                operand_d   = operation_d->AddSectionDictionary("OPERAND");
                argument_d->SetValue("TYPE", utils::etype_to_ctype_text(block.scope[tac.in2].type));
                operand_d->SetValue("TYPE",  utils::etype_to_ctype_text(block.scope[tac.in2].type));

                argument_d->SetIntValue("NR", tac.in2);
                operand_d->SetIntValue("NR", tac.in2);

                if (CONSTANT != block.scope[tac.in2].layout) {
                    argument_d->ShowSection("ARRAY");
                    operand_d->ShowSection("ARRAY");
                }
            case 2:
                operation_d->SetValue("NR_FINPUT", to_string(tac.in1));  // Not all have
                operator_d->SetIntValue("NR_FINPUT", tac.in1);

                argument_d  = kernel_d.AddSectionDictionary("ARGUMENT");
                operand_d   = operation_d->AddSectionDictionary("OPERAND");

                argument_d->SetValue("TYPE", utils::etype_to_ctype_text(block.scope[tac.in1].type));
                operand_d->SetValue("TYPE", utils::etype_to_ctype_text(block.scope[tac.in1].type));

                argument_d->SetIntValue("NR", tac.in1);
                operand_d->SetIntValue("NR", tac.in1);

                if (CONSTANT != block.scope[tac.in1].layout) {
                    argument_d->ShowSection("ARRAY");
                    operand_d->ShowSection("ARRAY");
                }
            case 1:
                argument_d = kernel_d.AddSectionDictionary("ARGUMENT");
                argument_d->SetValue("TYPE", utils::etype_to_ctype_text(block.scope[tac.out].type));
                argument_d->SetIntValue("NR", tac.out);
                argument_d->ShowSection("ARRAY");

                operation_d->SetValue("NR_OUTPUT", to_string(tac.out));
                operator_d->SetIntValue("NR_OUTPUT", tac.out);

                operand_d = operation_d->AddSectionDictionary("OPERAND");
                operand_d->SetValue("TYPE", utils::etype_to_ctype_text(block.scope[tac.out].type));
                operand_d->SetIntValue("NR", tac.out);
                operand_d->ShowSection("ARRAY");
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

}}}
