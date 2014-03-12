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

Specializer::~Specializer()
{
    DEBUG("++ Specializer::~Specializer()");
    DEBUG("-- Specializer::~Specializer()");
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
                    tpl_opcode  = "random.";
                    tpl_ndim    = "1d.";
                    break;
                case RANGE:
                    tpl_opcode = "range.";
                    tpl_ndim    = "1d.";
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
    return specialize(block, optimized, 0, block.length-1);
}

string Specializer::specialize(Block& block, bool optimized, size_t tac_start, size_t tac_end)
{
    string sourcecode  = "";

    ctemplate::TemplateDictionary kernel_d("KERNEL");   // Kernel - function wrapping code
    kernel_d.SetValue("SYMBOL", block.symbol);
    kernel_d.SetValue("SYMBOL_TEXT", block.symbol_text);

    for(size_t i=tac_start; i<=tac_end; ++i) {
        
        //
        // Grab the tac for which to generate sourcecode
        tac_t& tac = block.program[i];

        //
        // Skip code generation for system and extensions
        if ((tac.op == SYSTEM) || (tac.op == EXTENSION)) {
            continue;
        }

        //
        // The operation (ewise, reduction, scan, random, range).
        ctemplate::TemplateDictionary* operation_d  = kernel_d.AddIncludeDictionary("OPERATIONS");
        operation_d->SetFilename(template_filename(block, i, optimized));

        //
        // Reduction and scan specific expansions
        if ((tac.op == REDUCE) || (tac.op == SCAN)) {
            operation_d->SetValue("TYPE_OUTPUT", utils::etype_to_ctype_text(block.scope[tac.out].etype));
            operation_d->SetValue("TYPE_INPUT",  utils::etype_to_ctype_text(block.scope[tac.in1].etype));
            operation_d->SetValue("TYPE_AXIS",  "int64_t");
            if (tac.oper == ADD) {
                operation_d->SetIntValue("NEUTRAL_ELEMENT", 0);
            } else if (tac.oper == MULTIPLY) {
                operation_d->SetIntValue("NEUTRAL_ELEMENT", 1);
            }
        }

        ctemplate::TemplateDictionary* operator_d   = operation_d->AddSectionDictionary("OPERATORS");
        ctemplate::TemplateDictionary* argument_d;  // Block arguments
        ctemplate::TemplateDictionary* operand_d;   // Operator operands

        //
        // The operator +, -, /, min, max, sin, sqrt, etc...
        //        
        operator_d->SetValue("OPERATOR", cexpression(block, i));

        //
        //  The arguments / operands
        switch(utils::tac_noperands(tac)) {
            case 3:
                operation_d->SetIntValue("NR_SINPUT", tac.in2);  // Not all have
                operator_d->SetIntValue("NR_SINPUT", tac.out);
                argument_d  = kernel_d.AddSectionDictionary("ARGUMENT");
                operand_d   = operation_d->AddSectionDictionary("OPERAND");
                argument_d->SetValue("TYPE", utils::etype_to_ctype_text(block.scope[tac.in2].etype));
                operand_d->SetValue("TYPE",  utils::etype_to_ctype_text(block.scope[tac.in2].etype));

                argument_d->SetIntValue("NR", tac.in2);
                operand_d->SetIntValue("NR", tac.in2);

                if (CONSTANT != block.scope[tac.in2].layout) {
                    argument_d->ShowSection("ARRAY");
                    operand_d->ShowSection("ARRAY");
                }
            case 2:
                operation_d->SetIntValue("NR_FINPUT", tac.in1);  // Not all have
                operator_d->SetIntValue("NR_FINPUT", tac.in1);

                argument_d  = kernel_d.AddSectionDictionary("ARGUMENT");
                operand_d   = operation_d->AddSectionDictionary("OPERAND");

                argument_d->SetValue("TYPE", utils::etype_to_ctype_text(block.scope[tac.in1].etype));
                operand_d->SetValue("TYPE", utils::etype_to_ctype_text(block.scope[tac.in1].etype));

                argument_d->SetIntValue("NR", tac.in1);
                operand_d->SetIntValue("NR", tac.in1);

                if (CONSTANT != block.scope[tac.in1].layout) {
                    argument_d->ShowSection("ARRAY");
                    operand_d->ShowSection("ARRAY");
                }
            case 1:
                argument_d = kernel_d.AddSectionDictionary("ARGUMENT");
                argument_d->SetValue("TYPE", utils::etype_to_ctype_text(block.scope[tac.out].etype));
                argument_d->SetIntValue("NR", tac.out);
                argument_d->ShowSection("ARRAY");

                operation_d->SetIntValue("NR_OUTPUT", tac.out);
                operator_d->SetIntValue("NR_OUTPUT", tac.out);

                operand_d = operation_d->AddSectionDictionary("OPERAND");
                operand_d->SetValue("TYPE", utils::etype_to_ctype_text(block.scope[tac.out].etype));
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
