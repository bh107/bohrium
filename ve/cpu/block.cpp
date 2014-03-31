#include "block.hpp"

using namespace std;
namespace bohrium{
namespace engine{
namespace cpu{

const char Block::TAG[] = "Block";

Block::Block(bh_ir& ir, bh_dag& dag) : noperands(0), omask(0), ir(ir), dag(dag)
{
    size_t ps = (size_t)dag.nnode;
    if (ps<1) {
        fprintf(stderr, "This block is the empty program! You should not have called this!");
    }
    scope    = (operand_t*)malloc((1+3)*ps*sizeof(operand_t));
    program  = (tac_t*)malloc(ps*sizeof(tac_t));
    instr    = (bh_instruction**)malloc(ps*sizeof(bh_instruction*));
    length   = ps;
}

Block::~Block()
{
    DEBUG(TAG, "~Block() ++");
    free(scope);
    free(program);
    free(instr);
    DEBUG(TAG, "~Block() --");
}

bh_dag& Block::get_dag()
{
    return this->dag;
}

string Block::scope_text(string prefix)
{
    stringstream ss;
    ss << prefix << "scope {" << endl;
    for(size_t i=1; i<=noperands; ++i) {
        ss << prefix << "  [" << i << "]{";
        ss << " layout(" << utils::layout_text(scope[i].layout) << "),";
        ss << " nelem(" << scope[i].nelem << "),";
        ss << " data(" << *(scope[i].data) << "),";
        ss << " const_data(" << scope[i].const_data << "),";
        ss << " etype(" << utils::etype_text(scope[i].etype) << "),";
        ss << " ndim(" << scope[i].ndim << "),";
        ss << " start(" << scope[i].start << "),";        
        ss << " shape(";
        for(int64_t dim=0; dim < scope[i].ndim; ++dim) {
            ss << scope[i].shape[dim];
            if (dim != (scope[i].ndim-1)) {
                ss << prefix << ", ";
            }
        }
        ss << "),";
        ss << " stride(";
        for(int64_t dim=0; dim < scope[i].ndim; ++dim) {
            ss << scope[i].stride[dim];
            if (dim != (scope[i].ndim-1)) {
                ss << prefix << ", ";
            }
        }
        ss << ")";
        ss << "}" << endl;
    }
    ss << prefix << "}" << endl;

    return ss.str();
}

string Block::scope_text()
{
    return scope_text("");
}

string Block::text(std::string prefix)
{
    stringstream ss;
    ss << prefix;
    ss << "block(";
    ss << "length="       << length;
    ss << ", noperands="  << noperands;
    ss << ", omask="      << omask;
    ss << ") {"           << endl;
    ss << prefix << "  symbol(" << symbol << ")" << endl;
    ss << prefix << "  symbol_text(" << symbol_text << ")" << endl;

    ss << prefix << "  program {" << endl;
    for(size_t i=0; i<length; ++i) {
        ss << prefix << "    [" << i << "]" << utils::tac_text(program[i]) << endl;
    }
    ss << prefix << "  }" << endl;

    ss << scope_text(prefix+"  ");
    ss << prefix << "}";
    
    return ss.str();
}

string Block::text()
{
    return text("");
}

/**
 *  Create a symbol for the block.
 *
 *  The textual version of the  symbol looks something like::
 *  
 *  symbol_text = ZIP-ADD-2D~1~2~3_~1Cf~2Cf~3Cf
 *
 *  Which will be hashed to some uint32_t value::
 *
 *  symbol = 2111321312412321432424
 *
 *  NOTE: System and extension operations are ignored.
 *        If a block consists of nothing but system and/or extension
 *        opcodes then the symbol will be the empty string "".
 */
bool Block::symbolize()
{   
    DEBUG(TAG,"symbolize(void) : length("<< length << ")");
    bool symbolize_res = symbolize(0, length-1);
    DEBUG(TAG,"symbolize(void) : symbol("<< symbol << "), symbol_text("<< symbol_text << ");");
    return symbolize_res;
}

bool Block::symbolize(size_t tac_start, size_t tac_end)
{
    stringstream tacs,
                 operands;

    DEBUG(TAG,"symbolize("<< tac_start << ", " << tac_end << ")");

    //
    // Scope
    for(size_t i=1; i<=noperands; ++i) {
        operands << "~" << i;
        operands << utils::layout_text_shand(scope[i].layout);
        operands << utils::etype_text_shand(scope[i].etype);
    }

    //
    // Program
    bool first = true;
    for (size_t i=tac_start; i<=tac_end; ++i) {
        tac_t& tac = this->program[i];
       
        // Do not include system opcodes in the kernel symbol.
        if ((tac.op == SYSTEM) || (tac.op == EXTENSION)) {
            continue;
        }
        if (!first) {   // Separate op+oper with "_"
            tacs  << "_";
        }
        first = false;

        tacs << utils::operation_text(tac.op);
        tacs << "-" << utils::operator_text(tac.oper);
        tacs << "-";
        DEBUG(TAG, "symbolize(...) : tac.out.ndim(" << scope[tac.out].ndim << ")");
        size_t ndim = (tac.op == REDUCE) ? scope[tac.in1].ndim : scope[tac.out].ndim;
        if (ndim <= 3) {
            tacs << ndim;
        } else {
            tacs << "N";
        }
        tacs << "D";
        
        switch(utils::tac_noperands(tac)) {
            case 3:
                tacs << "~" << tac.out;
                tacs << "~" << tac.in1;
                tacs << "~" << tac.in2;
                break;

            case 2:
                tacs << "~" << tac.out;
                tacs << "~" << tac.in1;
                break;

            case 1:
                tacs << "~" << tac.out;
                break;

            case 0:
                break;

            default:
                fprintf(stderr, "Something horrible happened...\n");
        }
    }

    symbol_text = tacs.str() +"_"+ operands.str();
    symbol      = utils::hash_text(symbol_text);

    DEBUG(TAG,"symbolize(...) : symbol("<< symbol << "), symbol_text("<< symbol_text << ");");
    return true;
}

/**
 *  Add instruction operand as argument to block.
 *
 *  Reuses operands of equivalent meta-data.
 *
 *  @param instr        The instruction whos operand should be converted.
 *  @param operand_idx  Index of the operand to represent as arg_t
 *  @param block        The block in which scope the argument will exist.
 */
size_t Block::add_operand(bh_instruction& instr, size_t operand_idx)
{
    size_t arg_idx = ++(noperands);
    if (bh_is_constant(&instr.operand[operand_idx])) {
        scope[arg_idx].const_data   = &(instr.constant.value);
        scope[arg_idx].data         = &scope[arg_idx].const_data;
        scope[arg_idx].etype        = utils::bhtype_to_etype(instr.constant.type);
        scope[arg_idx].nelem        = 1;
        scope[arg_idx].ndim         = 1;
        scope[arg_idx].start        = 0;
        scope[arg_idx].shape        = instr.operand[operand_idx].shape;
        scope[arg_idx].shape[0]     = 1;
        scope[arg_idx].stride       = instr.operand[operand_idx].shape;
        scope[arg_idx].stride[0]    = 0;
        scope[arg_idx].layout       = CONSTANT;
    } else {
        scope[arg_idx].const_data= NULL;
        scope[arg_idx].data     = &(bh_base_array(&instr.operand[operand_idx])->data);
        scope[arg_idx].etype    = utils::bhtype_to_etype(bh_base_array(&instr.operand[operand_idx])->type);
        scope[arg_idx].nelem    = bh_base_array(&instr.operand[operand_idx])->nelem;
        scope[arg_idx].ndim     = instr.operand[operand_idx].ndim;
        scope[arg_idx].start    = instr.operand[operand_idx].start;
        scope[arg_idx].shape    = instr.operand[operand_idx].shape;
        scope[arg_idx].stride   = instr.operand[operand_idx].stride;

        if (utils::is_contiguous(scope[arg_idx])) {
            scope[arg_idx].layout = CONTIGUOUS;
        } else {
            scope[arg_idx].layout = STRIDED;
        }
    }

    //
    // Reuse operand identifiers: Detect if we have seen it before and reuse the name.
    if (NULL!=*(scope[arg_idx].data)) {
        for(size_t i=0; i<arg_idx; ++i) {
            if (!utils::equivalent_operands(scope[i], scope[arg_idx])) {
                continue; // Not equivalent, continue search.
            }
            // Found one! Use it instead of the incremented identifier.
            --noperands;
            arg_idx = i;
            break;
        }
    }
    return arg_idx;
}

}}}
