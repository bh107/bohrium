#include "specializer.hpp"
#include <set>

using namespace std;
namespace bohrium {
namespace engine {
namespace cpu {

const char Specializer::TAG[] = "Specializer";

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
    DEBUG(TAG, "~Specializer()");
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
    DEBUG(TAG,"specialize(..., " << tac_start << ", " << tac_end << ")");
    string sourcecode  = "";

    ctemplate::TemplateDictionary kernel_d("KERNEL");   // Kernel - function wrapping code
    kernel_d.SetValue("SYMBOL", block.symbol);
    kernel_d.SetValue("SYMBOL_TEXT", block.symbol_text);

    kernel_d.SetValue("MODE", (apply_fusion ? "FUSED" : "SIJ"));
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
    //apply_fusion = false;
    //
    // Now process the array operations
    for(size_t i=tac_start; i<=tac_end; ++i) {

        //
        // Skip code generation for system and extensions
        if ((block.program[i].op == SYSTEM) || (block.program[i].op == EXTENSION)) {
            continue;
        }
        
        //
        // Fusion setup
        //
        
        //
        // Basic fusion approach; count the amount of ops and create a range
        size_t  fuse_ops    = 0,
                fuse_start  = i,
                fuse_end    = i;

        if (apply_fusion) {
            //
            // The first operation in a potential range of fusable operations
            tac_t& first = block.program[i];

            //
            // Examine potential expansion of the range of fusable operations
            for(size_t j=i; (apply_fusion) && (j<=tac_end); ++j) {
                tac_t& next = block.program[j];
                if (next.op == SYSTEM) {   // Ignore
                    cout << "Ignoring sstem operation." << endl;
                    continue;
                }
                if (next.op == EXTENSION) {
                    cout << "WE GOT AN EXTENSION!!!!" << endl;
                    break;
                }
                if (!((next.op == ZIP) || (next.op == MAP))) {
                    cout << "Incompatible operation " << utils::operation_text(next.op) << endl;
                    break;
                }
                // At this point the operation is an array operation
                bool compat_operands = true;
                
                switch(utils::tac_noperands(next)) {
                    case 3:
                        compat_operands = compat_operands && (utils::compatible_operands(block.scope[first.out], block.scope[next.in2]));
                    case 2:
                        compat_operands = compat_operands && (utils::compatible_operands(block.scope[first.out], block.scope[next.in1]));
                        compat_operands = compat_operands && (utils::compatible_operands(block.scope[first.out], block.scope[next.out]));
                    break;

                    default:
                        fprintf(stderr, "ARGGG!!!!\n");
                }
                if (!compat_operands) {
                    break;
                }

                if (fuse_ops == 0) {    // First
                    fuse_start  = j;
                    fuse_end    = j;
                } else {                // Some point later
                    fuse_end = j;
                }
                fuse_ops++;
            }
        }

        //
        // Assign information needed for generation of operation and operator code
        ctemplate::TemplateDictionary* operation_d  = kernel_d.AddIncludeDictionary("OPERATIONS");
        operation_d->SetFilename(template_filename(block, fuse_start));

        set<size_t> operands;
        set<size_t>::iterator operands_it;

        cout << "FOPS " << fuse_ops << " START " << fuse_start << " END " << fuse_end << endl;
        for(i=fuse_start; i<=fuse_end; ++i) {

            tac_t& tac = block.program[i];
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

            //
            // The operator +, -, /, min, max, sin, sqrt, etc...
            //        
            ctemplate::TemplateDictionary* operator_d = operation_d->AddSectionDictionary("OPERATORS");
            operator_d->SetValue("OPERATOR", cexpression(block, i));

            //
            //  The arguments / operands
            switch(utils::tac_noperands(tac)) {
                case 3:
                    operation_d->SetIntValue("NR_SINPUT", tac.in2);  // Not all have
                    operator_d->SetIntValue("NR_SINPUT", tac.in2);

                    operands.insert(tac.in2);

                case 2:
                    operation_d->SetIntValue("NR_FINPUT", tac.in1);  // Not all have
                    operator_d->SetIntValue("NR_FINPUT", tac.in1);

                    operands.insert(tac.in1);

                case 1:
                    operation_d->SetIntValue("NR_OUTPUT", tac.out);
                    operator_d->SetIntValue("NR_OUTPUT", tac.out);

                    operands.insert(tac.out);
            }
        }

        //
        // Assign operands to the operation, we use a set to avoid redeclaration.
        for(operands_it=operands.begin(); operands_it != operands.end(); operands_it++) {
            ctemplate::TemplateDictionary* operand_d = operation_d->AddSectionDictionary("OPERAND");
            operand_d->SetValue("TYPE",  utils::etype_to_ctype_text(block.scope[*operands_it].etype));
            operand_d->SetIntValue("NR", *operands_it);

            if (CONSTANT != block.scope[*operands_it].layout) {
                operand_d->ShowSection("ARRAY");
            }   
        }
        operands.clear();
        i = fuse_end;
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

    DEBUG(TAG,"specialize(...);");
    return sourcecode;
}

}}}
