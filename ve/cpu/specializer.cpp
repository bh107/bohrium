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
string Specializer::template_filename(const Block& block, size_t pc)
{
    string tpl_ndim   = "nd.",
           tpl_opcode,
           tpl_layout = "strided.";

    operand_t* symbol_table = block.symbol_table.table;
    tac_t& tac = block.program[pc];
    int ndim = (tac.op == REDUCE)         ? \
               symbol_table[tac.in1].ndim : \
               symbol_table[tac.out].ndim;

    LAYOUT layout_out = symbol_table[tac.out].layout, 
           layout_in1 = symbol_table[tac.in1].layout,
           layout_in2 = symbol_table[tac.in2].layout;

    switch (tac.op) {                    // OPCODE_SWITCH
        case MAP:

            tpl_opcode  = "ewise.";
            if (((layout_out & CONT_COMPATIBLE)>0) && \
                ((layout_in1 & CONT_COMPATIBLE)>0)
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
            if (((layout_out & CONT_COMPATIBLE)>0) && \
                ((layout_in1 & CONT_COMPATIBLE)>0) && \
                ((layout_in2 & CONT_COMPATIBLE)>0)
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
string Specializer::specialize(const Block& block, bool apply_fusion)
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
string Specializer::specialize(const Block& block, size_t tac_start, size_t tac_end, bool apply_fusion)
{
    DEBUG(TAG,"specialize(..., " << tac_start << ", " << tac_end << ")");
    string sourcecode  = "";

    operand_t* symbol_table = block.symbol_table.table;

    ctemplate::TemplateDictionary kernel_d("KERNEL");   // Kernel - function wrapping code
    kernel_d.SetValue("SYMBOL", block.symbol);
    kernel_d.SetValue("SYMBOL_TEXT", block.symbol_text);

    kernel_d.SetValue("MODE", (apply_fusion ? "FUSED" : "SIJ"));
    kernel_d.SetIntValue("NINSTR", block.length);
    kernel_d.SetIntValue("NARGS", block.noperands);

    //
    // Now process the array operations
    for(size_t tac_idx=tac_start; tac_idx<=tac_end; ++tac_idx) {

        //
        // Skip code generation for system and extensions
        if ((block.program[tac_idx].op == SYSTEM) || (block.program[tac_idx].op == EXTENSION)) {
            continue;
        }
        
        //
        // Fusion setup
        //
        
        //
        // Basic fusion approach; count the amount of ops and create a range
        size_t  fuse_ops    = 0,
                fuse_start  = tac_idx,
                fuse_end    = tac_idx;

        if (apply_fusion) {
            //
            // The first operation in a potential range of fusable operations
            tac_t& first = block.program[tac_idx];

            //
            // Examine potential expansion of the range of fusable operations
            for(size_t j=tac_idx; (apply_fusion) && (j<=tac_end); ++j) {
                tac_t& next = block.program[j];
                if (next.op == SYSTEM) {   // Ignore
                    DEBUG(TAG, "Ignoring sstem operation.");
                    continue;
                }
                if (next.op == EXTENSION) {
                    DEBUG(TAG, "WE GOT AN EXTENSION!!!!");
                    break;
                }
                if (!((next.op == ZIP) || (next.op == MAP))) {
                    DEBUG(TAG, "Incompatible operation " << utils::operation_text(next.op));
                    break;
                }
                // At this point the operation is an array operation
                bool compat_operands = true;
                
                switch(utils::tac_noperands(next)) {
                    case 3:
                        compat_operands = compat_operands && (utils::compatible(symbol_table[first.out], symbol_table[next.in2]));
                    case 2:
                        compat_operands = compat_operands && (utils::compatible(symbol_table[first.out], symbol_table[next.in1]));
                        compat_operands = compat_operands && (utils::compatible(symbol_table[first.out], symbol_table[next.out]));
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

            //
            // Replace temporary arrays with scalars. This is done by "converting" the array into a scalar.
            //
            // TODO: Check that it is actually only used within the fuse-range...
            if ((fuse_ops>1) && (block.symbol_table.temps.size()>0)) {
                for(size_t j=fuse_start; j<=fuse_end; ++j) {
                    tac_t& cur = block.program[j];
                    switch(utils::tac_noperands(cur)) {
                        case 3:
                            if (block.symbol_table.temps.find(cur.in2) != block.symbol_table.temps.end()) {
                                block.symbol_table.turn_scalar(cur.in2);
                            }
                        case 2:
                            if (block.symbol_table.temps.find(cur.in1) != block.symbol_table.temps.end()) {
                                block.symbol_table.turn_scalar(cur.in1);
                            }
                        case 1:
                            if (block.symbol_table.temps.find(cur.out) != block.symbol_table.temps.end()) {
                                block.symbol_table.turn_scalar(cur.out);
                            }
                            break;
                    }
                }
            }
        }

        //
        // Assign information needed for generation of operation and operator code
        ctemplate::TemplateDictionary* operation_d  = kernel_d.AddIncludeDictionary("OPERATIONS");
        operation_d->SetFilename(template_filename(block, fuse_start));

        set<size_t> operands;
        set<size_t>::iterator operands_it;

        DEBUG(TAG, "FOPS " << fuse_ops << " START " << fuse_start << " END " << fuse_end);
        for(tac_idx=fuse_start; tac_idx<=fuse_end; ++tac_idx) {

            tac_t& tac = block.program[tac_idx];
            //
            // Reduction and scan specific expansions
            if ((tac.op == REDUCE) || (tac.op == SCAN)) {
                operation_d->SetValue("TYPE_OUTPUT", utils::etype_to_ctype_text(symbol_table[tac.out].etype));
                operation_d->SetValue("TYPE_INPUT",  utils::etype_to_ctype_text(symbol_table[tac.in1].etype));
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
            operator_d->SetValue("OPERATOR", cexpression(block, tac_idx));

            //
            // Map the tac operands into block-scope
            switch(utils::tac_noperands(tac)) {
                case 3:
                    operation_d->SetIntValue("NR_SINPUT", block.operand_map.find(tac.in2)->second);  // Not all have
                    operator_d->SetIntValue("NR_SINPUT",  block.operand_map.find(tac.in2)->second);

                    operands.insert(block.operand_map.find(tac.in2)->second);

                case 2:
                    operation_d->SetIntValue("NR_FINPUT", block.operand_map.find(tac.in1)->second);  // Not all have
                    operator_d->SetIntValue("NR_FINPUT",  block.operand_map.find(tac.in1)->second);

                    operands.insert(block.operand_map.find(tac.in1)->second);

                case 1:
                    operation_d->SetIntValue("NR_OUTPUT", block.operand_map.find(tac.out)->second);
                    operator_d->SetIntValue("NR_OUTPUT",  block.operand_map.find(tac.out)->second);

                    operands.insert(block.operand_map.find(tac.out)->second);
            }
        }

        //
        // Assign operands to the operation, we use a set to avoid redeclaration.
        for(operands_it=operands.begin(); operands_it != operands.end(); operands_it++) {
            size_t opr_idx = *operands_it;
            if (0 == opr_idx) {
                fprintf(stderr, "THIS SHOULD NEVER MAPPEN! OPERAND 0 is used!\n");
            }
            const operand_t& operand = block.scope(opr_idx);

            ctemplate::TemplateDictionary* operand_d = operation_d->AddSectionDictionary("OPERAND");
            operand_d->SetValue("TYPE",  utils::etype_to_ctype_text(operand.etype));
            operand_d->SetIntValue("NR", opr_idx);

            if ((operand.layout & ARRAY_LAYOUT)>0) {
                operand_d->ShowSection("ARRAY");
            }   
        }
        operands.clear();
        tac_idx = fuse_end;
    }

    //
    //  Assign arguments for kernel operand unpacking
    for(size_t opr_idx=1; opr_idx<=block.noperands; ++opr_idx) {
        const operand_t& operand = block.scope(opr_idx);
        ctemplate::TemplateDictionary* argument_d = kernel_d.AddSectionDictionary("ARGUMENT");
        argument_d->SetIntValue("NR", opr_idx);
        argument_d->SetValue("TYPE", utils::etype_to_ctype_text(operand.etype));
        switch(operand.layout) {
            case CONSTANT:
                argument_d->ShowSection("CONSTANT");
                break;
            case SCALAR:
                argument_d->ShowSection("SCALAR");
                break;
            case CONTIGUOUS:
            case STRIDED:
            case SPARSE:
                argument_d->ShowSection("ARRAY");
                break;
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

    DEBUG(TAG,"specialize(...);");
    return sourcecode;
}

}}}
