#include "block.hpp"

using namespace std;
namespace bohrium{
namespace engine{
namespace cpu{

const char Block::TAG[] = "Block";

Block::Block(SymbolTable& symbol_table, const bh_ir& ir, size_t dag_idx)
: scope(NULL), noperands(0), omask(0), symbol_table(symbol_table), ir(ir), dag(ir.dag_list[dag_idx])
{
    size_t ps = (size_t)dag.nnode;
    if (ps<1) {
        fprintf(stderr, "This block is the empty program! You should not have called this!");
    }

    scope    = (operand_t**)malloc((1+3)*ps*sizeof(operand_t*));
    scope[0] = &symbol_table.table[0];  // Always point to the pseudo-operand.

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

const bh_dag& Block::get_dag()
{
    return this->dag;
}

string Block::scope_text(string prefix)
{
    stringstream ss;
    ss << prefix << "scope {" << endl;
    for(size_t i=1; i<=noperands; ++i) {
        operand_t& operand = *scope[i];
        ss << prefix;
        ss << "[" << i << "] {";
        ss << utils::operand_text(operand);
        ss << "}";
        ss << endl;
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
        operand_t& operand = *scope[i];

        operands << "~" << i;
        operands << utils::layout_text_shand(operand.layout);
        operands << utils::etype_text_shand(operand.etype);
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
        DEBUG(TAG, "symbolize(...) : tac.out.ndim(" << scope[tac.out]->ndim << ")");
        size_t ndim = (tac.op == REDUCE) ? scope[tac.in1]->ndim : scope[tac.out]->ndim;
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

size_t Block::add_operand(bh_instruction& instr, size_t operand_idx)
{
    //
    // Map operands through the SymbolTable
    size_t arg_symbol = symbol_table.map_operand(instr, operand_idx);

    //
    // Map operands into block-scope.
    size_t arg_idx = ++(noperands);

    //
    // TODO: Replace this with references instead of copies
    scope[arg_idx] = &symbol_table.table[arg_symbol];

    //
    // Reuse operand identifiers: Detect if we have seen it before and reuse the name.
    // This is done by comparing the currently investigated operand (arg_idx)
    // with all other operands in the current scope [1,arg_idx[
    // Do remember that 0 is is not a valid operand and we therefore index from 1.
    // Also we do not want to compare with selv, that is when i == arg_idx.
    for(size_t i=1; i<arg_idx; ++i) {
        if (!utils::equivalent(*scope[i], *scope[arg_idx])) {
            continue; // Not equivalent, continue search.
        }
        // Found one! Use it instead of the incremented identifier.
        --noperands;
        arg_idx = i;
        break;
    }
    return arg_idx;
}

}}}
