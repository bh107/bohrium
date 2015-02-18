#include "specializer.hpp"
#include <set>
#include <algorithm>

#include "codegen.hpp"

using namespace std;
namespace bohrium {
namespace engine {
namespace cpu {

const char Specializer::TAG[] = "Specializer";

Specializer::Specializer(const string template_directory)
: plaid_(), strip_mode(ctemplate::STRIP_BLANK_LINES), template_directory(template_directory)
{
    ctemplate::mutable_default_template_cache()->SetTemplateRootDirectory(template_directory);
    ctemplate::LoadTemplate("ewise.cont.nd.tpl",    strip_mode);
    ctemplate::LoadTemplate("ewise.strided.1d.tpl", strip_mode);
    ctemplate::LoadTemplate("ewise.strided.2d.tpl", strip_mode);
    ctemplate::LoadTemplate("ewise.strided.3d.tpl", strip_mode);
    ctemplate::LoadTemplate("ewise.strided.nd.tpl", strip_mode);
    ctemplate::LoadTemplate("kernel.tpl",   strip_mode);
    ctemplate::LoadTemplate("license.tpl",  strip_mode);
    ctemplate::LoadTemplate("reduce.strided.1d.tpl", strip_mode);
    ctemplate::LoadTemplate("reduce.strided.2d.tpl", strip_mode);
    ctemplate::LoadTemplate("reduce.strided.3d.tpl", strip_mode);
    ctemplate::LoadTemplate("reduce.strided.nd.tpl", strip_mode);
    ctemplate::LoadTemplate("scan.strided.1d.tpl", strip_mode);
    ctemplate::LoadTemplate("scan.strided.nd.tpl", strip_mode);
    ctemplate::mutable_default_template_cache()->Freeze();

    /*
    plaid_.add_from_file("kernel", "/home/safl/bohrium/ve/cpu/nonlogicT/kernel.tpl");
    plaid_.add_from_file("ewise.cont.nd", "/home/safl/bohrium/ve/cpu/nonlogicT/ewise.cont.nd.tpl");
    plaid_.add_from_file("ewise.strided.nd", "/home/safl/bohrium/ve/cpu/nonlogicT/ewise.strided.nd.tpl");
    plaid_.add_from_file("ewise.strided.1d", "/home/safl/bohrium/ve/cpu/nonlogicT/ewise.strided.1d.tpl");
    plaid_.add_from_file("ewise.strided.2d", "/home/safl/bohrium/ve/cpu/nonlogicT/ewise.strided.2d.tpl");
    plaid_.add_from_file("ewise.strided.3d", "/home/safl/bohrium/ve/cpu/nonlogicT/ewise.strided.3d.tpl");
    */
}

Specializer::~Specializer()
{
    ctemplate::mutable_default_template_cache()->ClearCache();
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

    const tac_t& tac = block.tac(pc);
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
                    tpl_opcode  = "ewise.";
                    tpl_layout  = "cont.";
                    tpl_ndim    = "nd.";
                    break;
                case RANGE:
                    tpl_opcode  = "ewise.";
                    tpl_layout  = "cont.";
                    tpl_ndim    = "nd.";
                    break;
                default:
                    fprintf(
                        stderr,
                        "Operator %s is not supported with operation %s\n",
                        core::operation_text(tac.op).c_str(),
                        core::operator_text(tac.oper).c_str()
                    );
            }
            break;

        default:
            printf("template_filename: Err=[Unsupported operation %d.]\n", tac.oper);
            throw runtime_error("template_filename: No template for opcode.");
    }

    return tpl_opcode + tpl_layout + tpl_ndim  + "tpl";
}

string Specializer::specialize( SymbolTable& symbol_table,
                                Block& block,
                                LAYOUT fusion_layout)
{
    
    //codegen::Kernel krnl_cgen(plaid_, block);
    //cout << krnl_cgen.generate_source() << endl;

    string sourcecode = "";

    ctemplate::TemplateDictionary kernel_d("KERNEL");   // Kernel - function wrapping code
    kernel_d.SetValue("SYMBOL",         block.symbol());
    kernel_d.SetValue("SYMBOL_TEXT",    block.symbol_text());

    kernel_d.SetValue("MODE", "FUSED");
    kernel_d.SetValue("LAYOUT", layout_text(fusion_layout)); 
    kernel_d.SetIntValue("NINSTR", block.ntacs());
    kernel_d.SetIntValue("NARRAY_INSTR", block.narray_tacs());
    kernel_d.SetIntValue("NARGS", block.noperands());

    //
    // Assign information needed for generation of operation and operator code
    ctemplate::TemplateDictionary* operation_d = kernel_d.AddIncludeDictionary("OPERATIONS");

    stringstream tpl_filename;
    tpl_filename << "ewise.";
    switch(fusion_layout) {
        case SCALAR_CONST:
        case SCALAR_TEMP:
        case SCALAR:
        case CONTIGUOUS:
            tpl_filename << "cont.";
            tpl_filename << "nd.";
            break;
        case STRIDED:
        case SPARSE:
            tpl_filename << "strided.";
            switch(symbol_table[block.array_tac(0).out].ndim) {
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

    set<size_t> operands;
    for(size_t tac_idx=0; tac_idx<block.narray_tacs(); ++tac_idx) {
        tac_t& tac = block.array_tac(tac_idx);

        //
        // The operator +, -, /, min, max, sin, sqrt, etc...
        //        
        ctemplate::TemplateDictionary* operator_d = operation_d->AddSectionDictionary("OPERATORS");
        operator_d->SetValue("OPERATOR", cexpression(symbol_table, block, tac_idx));
        
        //
        // Map the tac operands into block-scope
        switch(core::tac_noperands(tac)) {
            case 3:
                operation_d->SetIntValue("NR_SINPUT", block.global_to_local(tac.in2));
                operator_d->SetIntValue("NR_SINPUT",  block.global_to_local(tac.in2));

                operands.insert(block.global_to_local(tac.in2));

            case 2:
                operation_d->SetIntValue("NR_FINPUT", block.global_to_local(tac.in1));
                operator_d->SetIntValue("NR_FINPUT",  block.global_to_local(tac.in1));

                operands.insert(block.global_to_local(tac.in1));

            case 1:
                operation_d->SetIntValue("NR_OUTPUT", block.global_to_local(tac.out));
                operator_d->SetIntValue("NR_OUTPUT",  block.global_to_local(tac.out));

                operands.insert(block.global_to_local(tac.out));
        }

    }

    //
    // Assign operands to the operation, we use a set to avoid redeclaration within the operation.
    for(set<size_t>::iterator operands_it=operands.begin();
        operands_it != operands.end();
        operands_it++) {

        size_t opr_idx = *operands_it;
        const operand_t& operand = block.operand(opr_idx);

        ctemplate::TemplateDictionary* operand_d = operation_d->AddSectionDictionary("OPERAND");
        operand_d->SetIntValue("NR", opr_idx);
        operand_d->SetValue("TYPE",  core::etype_to_ctype_text(operand.etype));

        switch(operand.layout) {
            case SCALAR_TEMP:
                operand_d->ShowSection("SCALAR_TEMP");
                break;
            case SCALAR_CONST:
                operand_d->ShowSection("SCALAR_CONST");
                break;
            case SCALAR:
                operand_d->ShowSection("SCALAR");
                break;
            case CONTIGUOUS:
            case STRIDED:
            case SPARSE:
                operand_d->ShowSection("ARRAY");
                break;
        }

        //
        //  Assign arguments for kernel operand unpacking
        ctemplate::TemplateDictionary* argument_d = kernel_d.AddSectionDictionary("ARGUMENT");
        argument_d->SetIntValue("NR", opr_idx);
        argument_d->SetValue("TYPE", core::etype_to_ctype_text(operand.etype));
        switch(operand.layout) {
            case SCALAR_TEMP:
                argument_d->ShowSection("SCALAR_TEMP");
                break;
            case SCALAR:
                argument_d->ShowSection("SCALAR");
                break;
            case SCALAR_CONST:
                argument_d->ShowSection("SCALAR_CONST");
                break;
            case CONTIGUOUS:
            case STRIDED:
            case SPARSE:
                argument_d->ShowSection("ARRAY");
                break;
        }
    }

    kernel_d.SetIntValue("NARRAY_ARGS", operands.size());
    operands.clear();

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
                                Block& block,
                                size_t tac_start, size_t tac_end)
{
    //codegen::Kernel krnl_cgen(plaid_, block);
    //cout << krnl_cgen.generate_source() << endl;

    string sourcecode  = "";

    ctemplate::TemplateDictionary kernel_d("KERNEL");   // Kernel - function wrapping code
    kernel_d.SetValue("SYMBOL", block.symbol());
    kernel_d.SetValue("SYMBOL_TEXT", block.symbol_text());

    kernel_d.SetValue("MODE", "SIJ");
    kernel_d.SetValue("LAYOUT", "INSPECT_THE_GENERATED_CODE"); 
    kernel_d.SetIntValue("NINSTR", block.ntacs());
    kernel_d.SetIntValue("NARRAY_INSTR", block.narray_tacs());
    kernel_d.SetIntValue("NARGS", block.noperands());

    //
    //  Assign arguments for kernel operand unpacking
    for(size_t opr_idx=0; opr_idx<block.noperands(); ++opr_idx) {
        const operand_t& operand = block.operand(opr_idx);
        ctemplate::TemplateDictionary* argument_d = kernel_d.AddSectionDictionary("ARGUMENT");
        argument_d->SetIntValue("NR", opr_idx);
        argument_d->SetValue("TYPE", core::etype_to_ctype_text(operand.etype));
        switch(operand.layout) {
            case SCALAR:
                argument_d->ShowSection("SCALAR");
                break;
            case SCALAR_CONST:
                argument_d->ShowSection("SCALAR_CONST");
                break;
            case SCALAR_TEMP:
                argument_d->ShowSection("SCALAR_TEMP");
                break;
            case CONTIGUOUS:
            case STRIDED:
            case SPARSE:
                argument_d->ShowSection("ARRAY");
                break;
        }
    }

    //
    // Now process the array operations
    for(size_t tac_idx=tac_start; tac_idx<=tac_end; ++tac_idx) {

        //
        // Skip code generation for system and extensions
        if ((block.tac(tac_idx).op == SYSTEM) || (block.tac(tac_idx).op == EXTENSION)) {
            continue;
        }

        //
        // Assign information needed for generation of operation and operator code
        ctemplate::TemplateDictionary* operation_d  = kernel_d.AddIncludeDictionary("OPERATIONS");

        operation_d->SetFilename(template_filename(symbol_table, block, tac_idx));
        set<size_t> operands;
        set<size_t>::iterator operands_it;

        tac_t& tac = block.tac(tac_idx);
        //
        // Reduction and scan specific expansions
        if ((tac.op == REDUCE) || (tac.op == SCAN)) {
            operation_d->SetValue("TYPE_OUTPUT", core::etype_to_ctype_text(symbol_table[tac.out].etype));
            operation_d->SetValue("TYPE_INPUT",  core::etype_to_ctype_text(symbol_table[tac.in1].etype));
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
        switch(core::tac_noperands(tac)) {
            case 3:
                operation_d->SetIntValue("NR_SINPUT", block.global_to_local(tac.in2));
                operator_d->SetIntValue("NR_SINPUT",  block.global_to_local(tac.in2));

                operands.insert(block.global_to_local(tac.in2));

            case 2:
                operation_d->SetIntValue("NR_FINPUT", block.global_to_local(tac.in1));
                operator_d->SetIntValue("NR_FINPUT",  block.global_to_local(tac.in1));

                operands.insert(block.global_to_local(tac.in1));

            case 1:
                operation_d->SetIntValue("NR_OUTPUT", block.global_to_local(tac.out));
                operator_d->SetIntValue("NR_OUTPUT",  block.global_to_local(tac.out));

                operands.insert(block.global_to_local(tac.out));
        }

        //
        // Assign operands to the operation, we use a set to avoid redeclaration.
        for(operands_it=operands.begin(); operands_it != operands.end(); operands_it++) {
            size_t opr_idx = *operands_it;
            const operand_t& operand = block.operand(opr_idx);

            ctemplate::TemplateDictionary* operand_d = operation_d->AddSectionDictionary("OPERAND");
            operand_d->SetIntValue("NR", opr_idx);
            operand_d->SetValue("TYPE",  core::etype_to_ctype_text(operand.etype));

            switch(operand.layout) {
                case SCALAR_TEMP:
                    operand_d->ShowSection("SCALAR_TEMP");
                    break;
                case SCALAR_CONST:
                    operand_d->ShowSection("SCALAR_CONST");
                    break;
                case SCALAR:
                    operand_d->ShowSection("SCALAR");
                    break;
                case CONTIGUOUS:
                case STRIDED:
                case SPARSE:
                    operand_d->ShowSection("ARRAY");
                    break;
            }
        }
        kernel_d.SetIntValue("NARRAY_ARGS", operands.size()); // NOTE: This is faulty when tac_start-tac_end>0!
        operands.clear();
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

string Specializer::specialize( SymbolTable& symbol_table,
                                Block& block)
{
    return specialize(symbol_table, block, 0, (block.ntacs()-1));
}

}}}
