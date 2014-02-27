#include "block.hpp"

using namespace std;
namespace bohrium{
namespace engine{
namespace cpu{

Block::Block(bh_ir& ir, bh_dag& dag) : noperands(0), omask(0), ir(ir), dag(dag)
{
    scope    = (operand_t*)malloc(1+3*dag.nnode*sizeof(operand_t));
    program  = (tac_t*)malloc(dag.nnode*sizeof(tac_t));
    instr    = (bh_instruction**)malloc(dag.nnode*sizeof(bh_instruction*));
    length   = dag.nnode;
}

Block::~Block()
{
    free(scope);
    free(program);
    free(instr);
}

string Block::text()
{
    stringstream ss;
    ss << "block(";
    ss << "length=" << to_string(length);
    ss << ", noperands=" << noperands;
    ss << ", omask=" << omask;
    ss << ") {" << endl;
    for(size_t i=0; i<length; ++i) {
        ss << "  " << utils::tac_text(program[i]);
    }
    ss << "  (" << symbol << ")" << endl;
    ss << "}";
    
    return ss.str();
}

/**
 *  Create a symbol for the block.
 *
 *  NOTE: System and extension operations are ignored.
 *        If a block consists of nothing but system and/or extension
 *        opcodes then the symbol will be the empty string "".
 */
bool Block::symbolize(const bool optimized) {

    std::string symbol_op_oper, 
                symbol_tsig,
                symbol_layout,
                symbol_ndim;

        symbol   = "";

    for (size_t i=0; i<length; ++i) {
        tac_t& tac = program[i];
        
        // Do not include system opcodes in the kernel symbol.
        if ((tac.op == SYSTEM) || (tac.op == EXTENSION)) {
            continue;
        }
        
        symbol_op_oper  += "_";
        symbol_op_oper  += utils::operation_text(tac.op);
        symbol_op_oper  += to_string(tac.oper);
        symbol_tsig     += utils::tac_typesig_text(tac, scope);
        symbol_layout   += utils::tac_layout_text(tac, scope);
    
        size_t ndim = scope[tac.out].ndim;
        if (tac.op == REDUCE) {
            ndim = scope[tac.in1].ndim;
        }
        if (optimized && (ndim <= 3)) {        // Optimized
            symbol_ndim += to_string(ndim);
        } else {
            symbol_ndim += string("N");
        }
        symbol_ndim += "D";
    }

    if ((omask & (BUILTIN_ARRAY_OPS)) > 0) {
        symbol = "BH" + \
                        symbol_op_oper  + "_" +\
                        symbol_tsig     + "_" +\
                        symbol_layout   + "_" +\
                        symbol_ndim;
    }
    return true;
}

/**
 *  Add instruction operand as argument to block.
 *
 *  @param instr        The instruction whos operand should be converted.
 *  @param operand_idx  Index of the operand to represent as arg_t
 *  @param block        The block in which scope the argument will exist.
 */
size_t Block::add_operand(bh_instruction& instr, size_t operand_idx)
{
    size_t arg_idx = ++(noperands);
    if (bh_is_constant(&instr.operand[operand_idx])) {
        scope[arg_idx].layout    = CONSTANT;
        scope[arg_idx].data      = &(instr.constant.value);
        scope[arg_idx].type      = utils::bhtype_to_etype(instr.constant.type);
        scope[arg_idx].nelem     = 1;
    } else {
        if (utils::is_contiguous(scope[arg_idx])) {
            scope[arg_idx].layout = CONTIGUOUS;
        } else {
            scope[arg_idx].layout = STRIDED;
        }
        scope[arg_idx].data      = bh_base_array(&instr.operand[operand_idx])->data;
        scope[arg_idx].type      = utils::bhtype_to_etype(bh_base_array(&instr.operand[operand_idx])->type);
        scope[arg_idx].nelem     = bh_base_array(&instr.operand[operand_idx])->nelem;
        scope[arg_idx].ndim      = instr.operand[operand_idx].ndim;
        scope[arg_idx].start     = instr.operand[operand_idx].start;
        scope[arg_idx].shape     = instr.operand[operand_idx].shape;
        scope[arg_idx].stride    = instr.operand[operand_idx].stride;
    }
    return arg_idx;
}

}}}
