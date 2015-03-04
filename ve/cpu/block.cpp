#include "block.hpp"
#include <iomanip>

using namespace std;
using namespace boost;

namespace bohrium{
namespace core{

const char Block::TAG[] = "Block";

Block::Block(SymbolTable& globals, vector<tac_t>& program)
: globals_(globals), program_(program), operands_(NULL), noperands_(0), symbol_text_(""), symbol_(""), omask_(0)
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
{                               // Reset the current state of the block
    tacs_.clear();              // tacs
    array_tacs_.clear();        // array_tacs

    omask_ = 0;

    iterspace_.layout = SCALAR; // iteraton space
    iterspace_.ndim = 0;
    iterspace_.shape = NULL;
    iterspace_.nelem = 0;
    
    if (operands_) {            // operands
        delete[] operands_;
        operands_   = NULL;
        noperands_  = 0;
    }
    global_to_local_.clear();   // global to local operand mapping
    local_to_global_.clear();   // local to global operand mapping

    symbol_text_    = "";       // textual symbol representation
    symbol_         = "";       // hashed symbol representation
}

void Block::compose(bh_ir_kernel& krnl)
{
    // An array pointers to operands
    // Will be handed to the kernel-function.
    operands_ = new operand_t*[krnl.instr_indexes.size()*3];

    for(std::vector<uint64_t>::iterator idx_it = krnl.instr_indexes.begin();
        idx_it != krnl.instr_indexes.end();
        ++idx_it) {

        tac_t& tac = program_[*idx_it];

        tacs_.push_back(&tac);              // <-- All tacs
        omask_ |= tac.op;                   // Update omask

        if ((tac.op & (ARRAY_OPS))>0) { 
            array_tacs_.push_back(&tac);    // <-- Only array operations
        }

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


void Block::compose(size_t prg_begin, size_t prg_end)
{
    // An array pointers to operands
    // Will be handed to the kernel-function.
    operands_ = new operand_t*[(prg_end-prg_begin+1)*3];

    for(size_t prg_idx=prg_begin; prg_idx<=prg_end; ++prg_idx) {
        tac_t& tac = program_[prg_idx];

        tacs_.push_back(&tac);              // <-- All tacs
        omask_ |= tac.op;                   // Update omask

        if ((tac.op & (ARRAY_OPS))>0) { 
            array_tacs_.push_back(&tac);    // <-- Only array operations
        }

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
    local_to_global_.insert(pair<size_t,size_t>(local_idx, global_idx));

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

size_t Block::noperands(void) const
{
    return noperands_;
}

size_t Block::global_to_local(size_t global_idx) const
{
    return global_to_local_.find(global_idx)->second;
}

size_t Block::local_to_global(size_t local_idx) const
{
    return local_to_global_.find(local_idx)->second;
}

tac_t& Block::tac(size_t idx) const
{
    return *tacs_[idx];
}

tac_t& Block::array_tac(size_t idx) const
{
    return *array_tacs_[idx];
}

uint32_t Block::omask(void)
{
    return omask_;
}

size_t Block::ntacs(void) const
{
    return tacs_.size();
}

size_t Block::narray_tacs(void) const
{
    return array_tacs_.size();
}

string Block::symbol(void) const
{
    return symbol_;
}

string Block::symbol_text(void) const
{
    return symbol_text_;
}

iterspace_t& Block::iterspace(void)
{
    return iterspace_;
}

void Block::update_iterspace(void)
{   
    // NOTE: This stuff is only valid for SIJ, and FUSION, streaming
    //       will require another way to determine the iteration space.
    
    //
    // Determine layout, ndim and shape
    for(size_t tac_idx=0; tac_idx<ntacs(); ++tac_idx) {
        tac_t& tac = this->tac(tac_idx);
        if (not ((tac.op & (ARRAY_OPS))>0)) {   // Only interested in array ops
            continue;
        } else if ((tac.op & (REDUCE))>0) {     // Reductions are weird
            if (globals_[tac.in1].layout > iterspace_.layout) {
                iterspace_.layout = globals_[tac.in1].layout;
                if ((iterspace_.layout & (ARRAY_LAYOUT))>0) {
                    iterspace_.ndim  = globals_[tac.in1].ndim;
                    iterspace_.shape = globals_[tac.in1].shape;
                }
            }
        } else {
            switch(tac_noperands(tac)) {        // Ewise are common-case
                case 3:
                    if (globals_[tac.in2].layout > iterspace_.layout) {
                        iterspace_.layout = globals_[tac.in2].layout;
                        if ((iterspace_.layout & (ARRAY_LAYOUT))>0) {
                            iterspace_.ndim  = globals_[tac.in2].ndim;
                            iterspace_.shape = globals_[tac.in2].shape;
                        }
                    }
                case 2:
                    if (globals_[tac.in1].layout > iterspace_.layout) {
                        iterspace_.layout = globals_[tac.in1].layout;
                        if ((iterspace_.layout & (ARRAY_LAYOUT))>0) {
                            iterspace_.ndim  = globals_[tac.in1].ndim;
                            iterspace_.shape = globals_[tac.in1].shape;
                        }
                    }
                case 1:
                    if (globals_[tac.out].layout > iterspace_.layout) {
                        iterspace_.layout = globals_[tac.out].layout;
                        if ((iterspace_.layout & (ARRAY_LAYOUT))>0) {
                            iterspace_.ndim  = globals_[tac.out].ndim;
                            iterspace_.shape = globals_[tac.out].shape;
                        }
                    }
                default:
                    break;
            }
        }
    }

    if (NULL != iterspace_.shape) {             // Determine number of elements
        iterspace_.nelem = 1;   
        for(int k=0; k<iterspace_.ndim; ++k) {
            iterspace_.nelem *= iterspace_.shape[k];
        }
    }
}

string Block::dot(void) const
{
    stringstream ss;
    return ss.str();
}

std::string Block::text_compact(void)
{
    stringstream ss;
    ss << setfill('0');
    ss << setw(3);
    ss << narray_tacs();
    ss << ",";
    ss << setfill(' ');
    ss << setw(36);
    ss << symbol();
    ss << ", ";
    ss << left;
    ss << setw(36);
    ss << setfill('-');
    ss << omask_aop_text(omask());
    ss << ", ";
    ss << left;
    ss << setfill('-');
    ss << setw(57);
    ss << iterspace_text(iterspace());

    return ss.str();
}

std::string Block::text(void)
{
    stringstream ss;
    ss << "BLOCK [" << endl;
    
    ss << "TACS (" << ntacs() << ") {" << endl;
    for(uint64_t tac_idx=0; tac_idx<ntacs(); ++tac_idx) {
        ss << tac_text(tac(tac_idx)) << endl;
    }
    ss << "}" << endl;

    ss << "OPERANDS (" << noperands() << ") {" << endl;
    for(size_t opr_idx=0; opr_idx < noperands(); ++opr_idx) {
        operand_t& opr = operand(opr_idx);
        ss << " loc_idx(" << opr_idx << ")";
        ss << " gbl_idx(" << local_to_global(opr_idx) << ") = ";
        ss << operand_text(opr);
    }
    ss << "}" << endl;

    ss << "ITERSPACE {" << endl;
    ss << " LAYOUT = " << layout_text(iterspace_.layout) << "," << endl;
    ss << " NDIM   = " << iterspace_.ndim << "," << endl;
    ss << " SHAPE  = {"; 
    for(int64_t dim=0; dim < iterspace_.ndim; ++dim) {
        ss << iterspace_.shape[dim];
        if (dim != (iterspace_.ndim-1)) {
            ss << ", ";
        }
    }
    ss << "}" << endl;
    ss << " NELEM  = " << iterspace_.nelem << endl;
    ss << "}" << endl;
    ss << "]" << endl;
    return ss.str();
}

}}
