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
string Specializer::template_filename(Block& block, size_t pc)
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
            if (
                (layout_out == CONTIGUOUS) && \
                ((layout_in1 == CONTIGUOUS) || (layout_out == CONSTANT))
               ) {
                tpl_layout  = "cont.";
            } else if (ndim == 1) {
                tpl_ndim = "1d.";
            } else if (ndim == 2) {
                tpl_ndim = "2d.";
            } else if (ndim == 3) {
                tpl_ndim = "3d.";
            }
            break;

        case ZIP:
            tpl_opcode  = "ewise.";
            if ( (layout_out == CONTIGUOUS) && \
                 (((layout_in1 == CONTIGUOUS) && (layout_in2 == CONTIGUOUS)) || \
                  ((layout_in1 == CONTIGUOUS) && (layout_in2 == CONSTANT)) || \
                  ((layout_in1 == CONSTANT) && (layout_in2 == CONTIGUOUS)) \
                 )
               ) {
                tpl_layout  = "cont.";
            } else if (ndim == 1) {
                tpl_ndim = "1d.";
            } else if (ndim == 2) {
                tpl_ndim = "2d.";
            } else if (ndim == 3) {
                tpl_ndim = "3d.";
            }
            break;

        case SCAN:
            tpl_opcode = "scan.";
            if (ndim == 1) {
                tpl_ndim = "1d.";
            }
            break;

        case REDUCE:
            tpl_opcode = "reduce.";
            if (ndim == 1) {
                tpl_ndim = "1d.";
            } else if (ndim == 2) {
                tpl_ndim = "2d.";
            } else if (ndim == 3) {
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
                    break;

                default:
                    fprintf(
                        stderr,
                        "Operator %s is not supported with operation %s\n",
                        utils::operation_text(tac.op).c_str(),
                        utils::operator_text(tac.oper).c_str()
                    );
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
 *  @param block The block to generate sourcecode for.
 *  @return The generated sourcecode.
 *
 */
string Specializer::specialize(Block& block, bool apply_fusion)
{
    return specialize(block, 0, block.length-1, apply_fusion);
}

/**
 *  Construct the c-sourcecode for the given block.
 *
 *  NOTE: System opcodes are ignored.
 *
 *  @param block The block to generate sourcecode for.
 *  @return The generated sourcecode.
 *
 */
string Specializer::specialize(Block& block, size_t tac_start, size_t tac_end, bool apply_fusion)
{
    DEBUG("Specializer::specialize(..., " << tac_start << ", " << tac_end << ")");
    string sourcecode  = "";

    ctemplate::TemplateDictionary kernel_d("KERNEL");   // Kernel - function wrapping code
    kernel_d.SetValue("SYMBOL", block.symbol);
    kernel_d.SetValue("SYMBOL_TEXT", block.symbol_text);

    kernel_d.SetValue("MODE", "SIJ");
    kernel_d.SetIntValue("NINSTR", block.length);
    kernel_d.SetIntValue("NARGS", block.noperands);

    //
    // Assign information needed for argument unpacking
    for(size_t i=1; i<=block.noperands; ++i) {
        ctemplate::TemplateDictionary* argument_d = kernel_d.AddSectionDictionary("ARGUMENT");
        argument_d->SetIntValue("NR", i);
        argument_d->SetValue("TYPE", utils::etype_to_ctype_text(block.scope[i].etype));
        if (CONSTANT != block.scope[i].layout) {
            argument_d->ShowSection("ARRAY");
        }
    }

    //
    // Assign information for needed for generation of operation and operator code
    ctemplate::TemplateDictionary* operation_d = NULL;
    int64_t prev_idx = -1;
    for(size_t i=tac_start; i<=tac_end; ++i) {
        
        //
        // Grab the tac for which to generate sourcecode
        tac_t& tac = block.program[i];

        //
        // Skip code generation for system and extensions
        if ((tac.op == SYSTEM) || (tac.op == EXTENSION)) {
            continue;
        }

        DEBUG("Specializer::specialize(...) : tac.out->ndim(" << block.scope[tac.out].ndim << ")");

        //
        // Basic fusability-check
        bool fusable = false;
        if (apply_fusion) {
            if (prev_idx >=0) {
                tac_t& prev = block.program[prev_idx];
                fusable = ( ((tac.op == MAP)    || (tac.op == ZIP))                         &&  \
                            ((prev.op == MAP)   || (prev.op == ZIP))                        &&  \
                            ((block.scope[tac.out].layout == block.scope[prev.out].layout)) &&  \
                            ((block.scope[tac.out].ndim == block.scope[prev.out].ndim))         \
                );
                if (fusable) {  // Check shape
                    for(int64_t dim=0; dim<block.scope[tac.out].ndim; ++dim) {
                        if (block.scope[tac.out].shape[dim] != block.scope[prev.out].shape[dim]) {
                            fusable = false;
                            break;
                        }
                    }
                }
            }
            prev_idx = i;
        }

        //
        // The operation (ewise, reduction, scan, random, range).
        if (!fusable) {
            operation_d  = kernel_d.AddIncludeDictionary("OPERATIONS");
            operation_d->SetFilename(template_filename(block, i));
        }

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

                operand_d   = operation_d->AddSectionDictionary("OPERAND");
                operand_d->SetValue("TYPE",  utils::etype_to_ctype_text(block.scope[tac.in2].etype));
                operand_d->SetIntValue("NR", tac.in2);

                if (CONSTANT != block.scope[tac.in2].layout) {
                    operand_d->ShowSection("ARRAY");
                }
            case 2:
                operation_d->SetIntValue("NR_FINPUT", tac.in1);  // Not all have
                operator_d->SetIntValue("NR_FINPUT", tac.in1);

                operand_d   = operation_d->AddSectionDictionary("OPERAND");
                operand_d->SetValue("TYPE", utils::etype_to_ctype_text(block.scope[tac.in1].etype));
                operand_d->SetIntValue("NR", tac.in1);

                if (CONSTANT != block.scope[tac.in1].layout) {
                    operand_d->ShowSection("ARRAY");
                }
            case 1:
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

    DEBUG("Specializer::specialize(...);");
    return sourcecode;
}

}}}
