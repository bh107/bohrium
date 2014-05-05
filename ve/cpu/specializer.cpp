#include "specializer.hpp"
#include <set>
#include <algorithm>

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
string Specializer::template_filename(SymbolTable& symbol_table, const Block& block, size_t pc)
{
    string tpl_ndim   = "nd.",
           tpl_opcode,
           tpl_layout = "strided.";

    const tac_t& tac = block.program(pc);
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
string Specializer::specialize( SymbolTable& symbol_table,
                                const Block& block)
{
    return specialize(symbol_table, block, 0, block.size()-1);
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
string Specializer::specialize( SymbolTable& symbol_table,
                                const Block& block,
                                size_t tac_start, size_t tac_end)
{
    DEBUG(TAG,"specialize(..., " << tac_start << ", " << tac_end << ")");
    string sourcecode  = "";

    ctemplate::TemplateDictionary kernel_d("KERNEL");   // Kernel - function wrapping code
    kernel_d.SetValue("SYMBOL", block.symbol());
    kernel_d.SetValue("SYMBOL_TEXT", block.symbol_text());

    kernel_d.SetValue("MODE", "SIJ");
    kernel_d.SetIntValue("NINSTR", block.size());
    kernel_d.SetIntValue("NARGS", block.noperands());

    //
    // Now process the array operations
    for(size_t tac_idx=tac_start; tac_idx<=tac_end; ++tac_idx) {

        //
        // Skip code generation for system and extensions
        if ((block.program(tac_idx).op == SYSTEM) || (block.program(tac_idx).op == EXTENSION)) {
            continue;
        }

        //
        // Assign information needed for generation of operation and operator code
        ctemplate::TemplateDictionary* operation_d  = kernel_d.AddIncludeDictionary("OPERATIONS");

        operation_d->SetFilename(template_filename(symbol_table, block, tac_idx));
        set<size_t> operands;
        set<size_t>::iterator operands_it;

        tac_t& tac = block.program(tac_idx);
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
        operator_d->SetValue("OPERATOR", cexpression(symbol_table, block, tac_idx));

        //
        // Map the tac operands into block-scope
        switch(utils::tac_noperands(tac)) {
            case 3:
                operation_d->SetIntValue("NR_SINPUT", block.resolve(tac.in2));
                operator_d->SetIntValue("NR_SINPUT",  block.resolve(tac.in2));

                operands.insert(block.resolve(tac.in2));

            case 2:
                operation_d->SetIntValue("NR_FINPUT", block.resolve(tac.in1));
                operator_d->SetIntValue("NR_FINPUT",  block.resolve(tac.in1));

                operands.insert(block.resolve(tac.in1));

            case 1:
                operation_d->SetIntValue("NR_OUTPUT", block.resolve(tac.out));
                operator_d->SetIntValue("NR_OUTPUT",  block.resolve(tac.out));

                operands.insert(block.resolve(tac.out));
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
    }

    //
    //  Assign arguments for kernel operand unpacking
    for(size_t opr_idx=1; opr_idx<=block.noperands(); ++opr_idx) {
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

/**
 *  Construct the c-sourcecode for the given block.
 *
 *  NOTE: System opcodes are ignored.
 *
 *  @param block The block to generate sourcecode for.
 *  @return The generated sourcecode.
 *
 */
string Specializer::specialize( SymbolTable& symbol_table,
                                const Block& block,
                                vector<triplet_t>& ranges)
{
    DEBUG(TAG,"specialize(..., ranges)");
    string sourcecode  = "";

    ctemplate::TemplateDictionary kernel_d("KERNEL");   // Kernel - function wrapping code
    kernel_d.SetValue("SYMBOL", block.symbol());
    kernel_d.SetValue("SYMBOL_TEXT", block.symbol_text());

    kernel_d.SetValue("MODE", "FUSED");
    kernel_d.SetIntValue("NINSTR", block.size());
    kernel_d.SetIntValue("NARGS", block.noperands());

    //
    // Now process the array operations
    //for(size_t tac_idx=tac_start; tac_idx<=tac_end; ++tac_idx) {
    for(vector<triplet_t>::iterator range_it = ranges.begin();
        range_it!=ranges.end();
        range_it++) {
        size_t range_begin  = (*range_it).begin;
        size_t range_end    = (*range_it).end;
        LAYOUT fusion_layout= (*range_it).layout;

        //
        // Find the first operation which is an array-op.
        size_t first_arrayop = 0;
        for(size_t tac_idx = range_begin; tac_idx<=range_end; tac_idx++) {
            if ((block.program(tac_idx).op & ARRAY_OPS)>0) {
                first_arrayop = tac_idx;
                break;
            }
        }
        //
        // If there arent any then we continue...
        if ((first_arrayop < range_begin) || (first_arrayop>range_end)) {
            continue;
        }

        //
        // Assign information needed for generation of operation and operator code
        ctemplate::TemplateDictionary* operation_d  = kernel_d.AddIncludeDictionary("OPERATIONS");

        if ((range_end - range_begin)>1) {
            stringstream tpl_filename;
            tpl_filename << "ewise.";
            switch(fusion_layout) {
                case CONSTANT:
                case SCALAR:
                case CONTIGUOUS:
                    tpl_filename << "cont.";
                    tpl_filename << "nd.";
                    break;
                case STRIDED:
                case SPARSE:
                    tpl_filename << "strided.";
                    switch(symbol_table[block.program(first_arrayop).out].ndim) {
                        case 3:
                            tpl_filename << "3d.";
                            break;
                        case 2:
                            tpl_filename << "2d.";
                            break;
                        case 1:
                            tpl_filename << "1d.";
                            break;
                        default:
                            tpl_filename << "nd.";
                    }
                    break;
            }
            tpl_filename << "tpl";
            operation_d->SetFilename(tpl_filename.str());
        } else {
            operation_d->SetFilename(template_filename(symbol_table, block, first_arrayop));
        }

        set<size_t> operands;
        set<size_t>::iterator operands_it;

        for(size_t tac_idx=range_begin; tac_idx<=range_end; tac_idx++) {
            tac_t& tac = block.program(tac_idx);
            if ((tac.op == SYSTEM) || (tac.op == EXTENSION)) {
                continue;
            }
            //
            // The operator +, -, /, min, max, sin, sqrt, etc...
            //        
            ctemplate::TemplateDictionary* operator_d = operation_d->AddSectionDictionary("OPERATORS");
            operator_d->SetValue("OPERATOR", cexpression(symbol_table, block, tac_idx));

            //
            // Map the tac operands into block-scope
            switch(utils::tac_noperands(tac)) {
                case 3:
                    operation_d->SetIntValue("NR_SINPUT", block.resolve(tac.in2));
                    operator_d->SetIntValue("NR_SINPUT",  block.resolve(tac.in2));

                    operands.insert(block.resolve(tac.in2));

                case 2:
                    operation_d->SetIntValue("NR_FINPUT", block.resolve(tac.in1));
                    operator_d->SetIntValue("NR_FINPUT",  block.resolve(tac.in1));

                    operands.insert(block.resolve(tac.in1));

                case 1:
                    operation_d->SetIntValue("NR_OUTPUT", block.resolve(tac.out));
                    operator_d->SetIntValue("NR_OUTPUT",  block.resolve(tac.out));

                    operands.insert(block.resolve(tac.out));
            }

        }

        tac_t& first_tac = block.program(first_arrayop);
        //
        // Reduction and scan specific expansions
        if ((first_tac.op == REDUCE) || (first_tac.op == SCAN)) {
            operation_d->SetValue("TYPE_OUTPUT", utils::etype_to_ctype_text(symbol_table[first_tac.out].etype));
            operation_d->SetValue("TYPE_INPUT",  utils::etype_to_ctype_text(symbol_table[first_tac.in1].etype));
            operation_d->SetValue("TYPE_AXIS",  "int64_t");
            if (first_tac.oper == ADD) {
                operation_d->SetIntValue("NEUTRAL_ELEMENT", 0);
            } else if (first_tac.oper == MULTIPLY) {
                operation_d->SetIntValue("NEUTRAL_ELEMENT", 1);
            }
        }

        //
        // Assign operands to the operation, we use a set to avoid redeclaration within the operation.
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
    }

    //
    //  Assign arguments for kernel operand unpacking
    for(size_t opr_idx=1; opr_idx<=block.noperands(); ++opr_idx) {
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
