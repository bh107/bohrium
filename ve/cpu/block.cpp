#include "block.hpp"

using namespace std;
using namespace boost;

namespace bohrium{
namespace core{

const char Block::TAG[] = "Block";

Block::Block(SymbolTable& globals, vector<tac_t>& program)
: globals_(globals), program_(program), operands_(NULL), noperands_(0), symbol_text_(""), symbol_("")
{}

Block::~Block()
{
    if (operands_) {
        delete[] operands_;
        operands_   = NULL;
        noperands_  = 0;
    }
}

void Block::clear(void)
{
    tacs_.clear();      // Reset the current state of the blocks
    
    if (operands_) {
        delete[] operands_;
        operands_   = NULL;
        noperands_  = 0;
    }
    global_to_local_.clear();

    symbol_text_    = "";
    symbol_         = "";
}

void Block::compose(size_t prg_begin, size_t prg_end)
{
    operands_ = new operand_t*[(prg_end-prg_begin+1)*3];

    for(size_t prg_idx=prg_begin; prg_idx<=prg_end; ++prg_idx) {
        tac_t& tac = program_[prg_idx];
        tacs_.push_back(&tac);

        // Map operands to local-scope
        switch(tac_noperands(tac)) {
            case 3:
                localize(tac.in2);
            case 2:
                localize(tac.in1);
            case 1:
                localize(tac.out);
            default:
                break;
        }
    }
}

void Block::compose(Graph& subgraph)
{
    operands_ = new operand_t*[(num_vertices(subgraph)+1)*3];

    // Fill tacs_ based on the subgraph
    std::pair<vertex_iter, vertex_iter> vip = vertices(subgraph);
    for(vertex_iter vi = vip.first; vi != vip.second; ++vi) {
        tac_t& tac = program_[subgraph.local_to_global(*vi)];
        tacs_.push_back(&tac);

        // Map operands to local-scope
        switch(tac_noperands(tac)) {
            case 3:
                localize(tac.in2);
            case 2:
                localize(tac.in1);
            case 1:
                localize(tac.out);
            default:
                break;
        }
    }
}

size_t Block::localize(size_t global_idx)
{
    //
    // Reuse operand identifiers: Detect if we have seen it before and reuse the index.
    size_t local_idx = 0;
    bool found = false;
    for(size_t i=0; i<noperands_; ++i) {
        if (!core::equivalent(*operands_[i], globals_[global_idx])) {
            continue; // Not equivalent, continue search.
        }
        // Found one! Use it instead of the incremented identifier.
        local_idx = i;
        found = true;
        break;
    }

    // Create the operand in block-scope
    if (!found) {
        local_idx = noperands_;
        operands_[local_idx] = &globals_[global_idx];
        ++noperands_;
    }

    //
    // Insert entry such that tac operands can be resolved in block-scope.
    global_to_local_.insert(pair<size_t,size_t>(global_idx, local_idx));

    return local_idx;
}

bool Block::symbolize(void)
{
    stringstream tacs, operands_ss;

    //
    // Scope
    for(size_t i=0; i<noperands_; ++i) {
        operands_ss << "~" << i;
        operands_ss << core::layout_text_shand(operands_[i]->layout);
        operands_ss << core::etype_text_shand(operands_[i]->etype);
    }

    //
    // Program
    bool first = true;
    for(vector<tac_t*>::iterator tac_iter=tacs_.begin(); tac_iter!=tacs_.end(); ++tac_iter) {
        tac_t& tac = **tac_iter;
       
        // Do not include system opcodes in the kernel symbol.
        if ((tac.op == SYSTEM) || (tac.op == EXTENSION)) {
            continue;
        }
        if (!first) {   // Separate op+oper with "_"
            tacs  << "_";
        }
        first = false;

        tacs << core::operation_text(tac.op);
        tacs << "-" << core::operator_text(tac.oper);
        tacs << "-";
        size_t ndim = (tac.op == REDUCE) ? globals_[tac.in1].ndim : globals_[tac.out].ndim;
        if (ndim <= 3) {
            tacs << ndim;
        } else {
            tacs << "N";
        }
        tacs << "D";
        
        switch(core::tac_noperands(tac)) {
            case 3:
                tacs << "_" << global_to_local(tac.out);
                tacs << "_" << global_to_local(tac.in1);
                tacs << "_" << global_to_local(tac.in2);
                break;

            case 2:
                tacs << "_" << global_to_local(tac.out);
                tacs << "_" << global_to_local(tac.in1);
                break;

            case 1:
                tacs << "_" << global_to_local(tac.out);
                break;

            case 0:
                break;

            default:
                fprintf(stderr, "Something horrible happened...\n");
        }
    }

    symbol_text_    = tacs.str() +"_"+ operands_ss.str();
    symbol_         = core::hash_text(symbol_text_);

    return true;
}

operand_t& Block::operand(size_t local_idx)
{
    return *operands_[local_idx];
}

operand_t** Block::operands(void)
{
    return operands_;
}

size_t Block::noperands(void)
{
    return noperands_;
}

size_t Block::global_to_local(size_t global_idx) const
{
    return global_to_local_.find(global_idx)->second;
}

tac_t& Block::tac(size_t idx) const
{
    return *tacs_[idx];
}

size_t Block::ntacs(void) const
{
    return tacs_.size();
}

string Block::symbol(void) const
{
    return symbol_;
}

string Block::symbol_text(void) const
{
    return symbol_text_;
}

}}
