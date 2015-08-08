#include "block.hpp"
#include <iomanip>

using namespace std;

namespace bohrium{
namespace core{

const char Block::TAG[] = "Block";

Block::Block(SymbolTable& globals, vector<kp_tac>& program)
: omask_(0), buffers_(NULL), nbuffers_(0), operands_(NULL), noperands_(0), globals_(globals), program_(program), symbol_text_(""), symbol_(""), footprint_nelem_(0), footprint_bytes_(0)
{}

Block::~Block()
{
    clear();
}

void Block::clear(void)
{                                   // Reset the current state of the block
    omask_ = 0;                     // Operation mask

    if (buffers_) {                 // Buffers
        delete[] buffers_;
        buffers_ = NULL;
        nbuffers_ = 0;
    }
    buffer_ids_.clear();
    input_buffers_.clear();
    output_buffers_.clear();
    buffer_refs_.clear();
    
    if (operands_) {                // Operands
        delete[] operands_;
        operands_   = NULL;
        noperands_  = 0;
    }
    global_to_local_.clear();   // global to local kp_operand mapping
    local_to_global_.clear();   // local to global kp_operand mapping

    iterspace_.layout = KP_SCALAR_TEMP;// Iteraton space
    iterspace_.ndim = 0;
    iterspace_.shape = NULL;
    iterspace_.nelem = 0;

    tacs_.clear();                  // tacs
    array_tacs_.clear();            // array_tacs

    symbol_text_    = "";       // textual symbol representation
    symbol_         = "";       // hashed symbol representation

    footprint_nelem_ = 0;
    footprint_bytes_ = 0;
}

void Block::_compose(bh_ir_kernel& krnl, bool array_contraction, size_t prg_idx)
{
    kp_tac & tac = program_[prg_idx];

    tacs_.push_back(&tac);              // <-- All tacs
    omask_ |= tac.op;                   // Update omask

    if ((tac.op & (KP_ARRAY_OPS))>0) {
        array_tacs_.push_back(&tac);    // <-- Only array operations
    }

    switch(tac_noperands(tac)) {
        case 3:
            _localize_scope(tac.in2);                   // Localize scope,      in2
            if ((tac.op & (KP_ARRAY_OPS))>0) {
                if (array_contraction && (krnl.get_temps().find((bh_base*)globals_[tac.in2].base)!=krnl.get_temps().end())) {
                    globals_.turn_contractable(tac.in2);    // Mark contractable,   in2
                }
                _bufferize(tac.in2);                        // Note down buffer id, in2
            }
        case 2:
            _localize_scope(tac.in1);                   // Localize scope,      in1
            if ((tac.op & (KP_ARRAY_OPS))>0) {
                if (array_contraction && (krnl.get_temps().find((bh_base*)globals_[tac.in1].base)!=krnl.get_temps().end())) {
                    globals_.turn_contractable(tac.in1);    // Mark contractable,   in1
                }
                _bufferize(tac.in1);                        // Note down buffer id, in1
            }
        case 1:
            _localize_scope(tac.out);                   // Localize scope,      out
            if ((tac.op & (KP_ARRAY_OPS))>0) {
                if (array_contraction && (krnl.get_temps().find((bh_base*)globals_[tac.out].base)!=krnl.get_temps().end())) {
                    globals_.turn_contractable(tac.out);    // Mark contractable,   out
                }
                _bufferize(tac.out);                        // Note down buffer id, out
            }
        default:
            break;
    }
}

void Block::compose(bh_ir_kernel& krnl, bool array_contraction)
{
    buffers_ = new kp_buffer *[krnl.instr_indexes.size()*3];
    operands_ = new kp_operand *[krnl.instr_indexes.size()*3];

    for(std::vector<uint64_t>::iterator idx_it = krnl.instr_indexes.begin();
        idx_it != krnl.instr_indexes.end();
        ++idx_it) {

        _compose(krnl, array_contraction, *idx_it);
    }

    if (array_contraction) {    // Turn kernel-temps into scalars aka array-contraction
        for (bh_base* base: krnl.get_temps()) {
            for(size_t operand_idx = 0;
                operand_idx < noperands();
                ++operand_idx) {
                if (operand(operand_idx).base == (kp_buffer*)base) {
                    globals_.turn_contractable(local_to_global(operand_idx));
                }
            }
        }
    }

    _update_iterspace();        // Update the iteration space
    // TODO: Classify buffers
}

void Block::compose(bh_ir_kernel& krnl, size_t prg_idx)
{
    buffers_ = new kp_buffer *[3];
    operands_ = new kp_operand *[3];
    
    _compose(krnl, false, prg_idx);
    _update_iterspace();                        // Update the iteration space
    // TODO: Classify buffers
}

void Block::_bufferize(size_t global_idx)
{
    // Maintain references to buffers within the block.
    if ((globals_[global_idx].layout & (KP_DYNALLOC_LAYOUT))>0) {
        kp_buffer * buffer = globals_[global_idx].base;

        std::map<kp_buffer *, size_t>::iterator buf = buffer_ids_.find(buffer);
        if (buf == buffer_ids_.end()) {
            size_t buffer_id = nbuffers_++;
            buffer_ids_.insert(pair<kp_buffer *, size_t>(
                buffer,
                buffer_id
            ));
            buffers_[buffer_id] = buffer;
        }

        buffer_refs_[buffer].insert(global_idx);
    }
} 

size_t Block::_localize_scope(size_t global_idx)
{
    //
    // Reuse kp_operand identifiers: Detect if we have seen it before and reuse the index.
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

    // Create the kp_operand in block-scope
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

        // Let the "Restrictable" flag be part of the symbol.
        if (buffer_refs_[operands_[i]->base].size()==1) {
            operands_ss << "R";
        } else {
            operands_ss << "A";
        }
    }

    //
    // Program
    bool first = true;
    for(vector<kp_tac *>::iterator tac_iter=tacs_.begin(); tac_iter!=tacs_.end(); ++tac_iter) {
        kp_tac & tac = **tac_iter;
       
        // Do not include system opcodes in the kernel symbol.
        if ((tac.op == KP_SYSTEM) || (tac.op == KP_EXTENSION)) {
            continue;
        }
        if (!first) {   // Separate op+oper with "_"
            tacs  << "_";
        }
        first = false;

        tacs << core::operation_text(tac.op);

        size_t ndim = ((tac.op & (KP_REDUCE_COMPLETE | KP_REDUCE_PARTIAL))>0) ? globals_[tac.in1].ndim : globals_[tac.out].ndim;

        //
        // Adding info of whether the kernel does reduction on the inner-most
        // dimensions or another "axis" dimension.
        //
        if ((tac.op & KP_REDUCE_PARTIAL)>0) {
            if (*((uint64_t*)globals_[tac.in2].const_data) == (ndim-1)) {
                tacs << "_INNER";
            } else {
                tacs << "_AXIS";
            }
        }

        //
        // Add ndim up to 3
        //
        tacs << "-" << core::operator_text(tac.oper);
        tacs << "-";
        if (ndim <= 3) {
            tacs << ndim;
        } else {
            tacs << "N";
        }
        tacs << "D";
        
        // Add operand IDs
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

kp_buffer & Block::buffer(size_t buffer_id)
{
    return *buffers_[buffer_id];
}

size_t Block::resolve_buffer(kp_buffer * buffer)
{
    std::map<kp_buffer *, size_t>::iterator buf = buffer_ids_.find(buffer);
    if (buf == buffer_ids_.end()) {
        // TODO: Raise exception
    }
    return buf->second;
}

kp_buffer ** Block::buffers(void)
{
    return buffers_;
}

size_t Block::nbuffers(void)
{
    return nbuffers_;
}

size_t Block::base_refcount(kp_buffer * base)
{
    return buffer_refs_[base].size();
}

kp_operand & Block::operand(size_t local_idx)
{
    return *operands_[local_idx];
}

kp_operand ** Block::operands(void)
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

kp_tac & Block::tac(size_t idx) const
{
    return *tacs_[idx];
}

kp_tac & Block::array_tac(size_t idx) const
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

kp_iterspace & Block::iterspace(void)
{
    return iterspace_;
}

void Block::_update_iterspace(void)
{       
    std::set<const kp_buffer *> footprint;

    //
    // Determine layout, ndim and shape
    for(size_t tac_idx=0; tac_idx<ntacs(); ++tac_idx) {
        kp_tac & tac = this->tac(tac_idx);
        if (not ((tac.op & (KP_ARRAY_OPS))>0)) {   // Only interested in array ops
            continue;
        }
        if ((tac.op & (KP_REDUCE_COMPLETE | KP_REDUCE_PARTIAL))>0) {     // Reductions are weird
            if (globals_[tac.in1].layout >= iterspace_.layout) {    // Iterspace
                iterspace_.layout = globals_[tac.in1].layout;
                iterspace_.ndim  = globals_[tac.in1].ndim;
                iterspace_.shape = globals_[tac.in1].shape;
            }
            if (globals_[tac.out].layout > iterspace_.layout) {
                iterspace_.layout = globals_[tac.out].layout;
            }
            if ((globals_[tac.out].layout & (KP_DYNALLOC_LAYOUT))>0) {   // Footprint
                footprint.insert(globals_[tac.out].base);
                output_buffers_.insert(globals_[tac.in1].base);
            }
            if ((globals_[tac.in1].layout & (KP_DYNALLOC_LAYOUT))>0) {
                footprint.insert(globals_[tac.in1].base);
                input_buffers_.insert(globals_[tac.in1].base);
            }
        } else {
            switch(tac_noperands(tac)) {
                case 3:
                    if (globals_[tac.in2].layout > iterspace_.layout) {     // Iterspace
                        iterspace_.layout = globals_[tac.in2].layout;
                        if (iterspace_.layout > KP_SCALAR_TEMP) {
                            iterspace_.ndim  = globals_[tac.in2].ndim;
                            iterspace_.shape = globals_[tac.in2].shape;
                        }
                    }
                    if ((globals_[tac.in2].layout & (KP_DYNALLOC_LAYOUT))>0) { // Footprint
                        footprint.insert(globals_[tac.in2].base);
                        input_buffers_.insert(globals_[tac.in2].base);
                    }

                case 2:
                    if (globals_[tac.in1].layout > iterspace_.layout) {     // Iterspace
                        iterspace_.layout = globals_[tac.in1].layout;
                        if (iterspace_.layout > KP_SCALAR_TEMP) {
                            iterspace_.ndim  = globals_[tac.in1].ndim;
                            iterspace_.shape = globals_[tac.in1].shape;
                        }
                    }
                    if ((globals_[tac.in1].layout & (KP_DYNALLOC_LAYOUT))>0) { // Footprint
                        footprint.insert(globals_[tac.in1].base);
                        input_buffers_.insert(globals_[tac.in1].base);
                    }

                case 1:
                    if (globals_[tac.out].layout > iterspace_.layout) {     // Iterspace
                        iterspace_.layout = globals_[tac.out].layout;
                        if (iterspace_.layout > KP_SCALAR_TEMP) {
                            iterspace_.ndim  = globals_[tac.out].ndim;
                            iterspace_.shape = globals_[tac.out].shape;
                        }
                    }
                    if ((globals_[tac.out].layout & (KP_DYNALLOC_LAYOUT))>0) { // Footprint
                        footprint.insert(globals_[tac.out].base);
                        output_buffers_.insert(globals_[tac.in1].base);
                    }

                default:
                    break;
            }
        }
    }

    footprint_nelem_ = 0;                       // Compute the footprint
    footprint_bytes_ = 0;
    for (std::set<const kp_buffer*>::iterator it=footprint.begin();
         it!=footprint.end();
         ++it) {
        footprint_nelem_ += (*it)->nelem;
        footprint_bytes_ += kp_buffer_nbytes(*it);
    }

    if (NULL != iterspace_.shape) {             // Determine number of elements
        iterspace_.nelem = 1;   
        for(int k=0; k<iterspace_.ndim; ++k) {
            iterspace_.nelem *= iterspace_.shape[k];
        }
    }
}

size_t Block::footprint_nelem(void)
{
    return footprint_nelem_;
}

size_t Block::footprint_bytes(void)
{
    return footprint_bytes_;
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
    ss << "BLOCK [" << symbol() << endl;
    ss << "TACS " << "omask(" << omask() << "), ntacs(" << ntacs() << ") {" << endl;
    for(uint64_t tac_idx=0; tac_idx<ntacs(); ++tac_idx) {
        ss << " " << tac_text(tac(tac_idx)) << endl;
    }
    ss << "}" << endl;

    ss << "OPERANDS (" << noperands() << ") {" << endl;
    for(size_t opr_idx=0; opr_idx < noperands(); ++opr_idx) {
        kp_operand & opr = operand(opr_idx);
        ss << " loc_idx(" << opr_idx << ")";
        ss << " gbl_idx(" << local_to_global(opr_idx) << ") = ";
        ss << operand_text(opr);
        ss << endl;
    }
    ss << "}" << endl;

    ss << "BASE_REFS {" << endl;
    for(std::map<kp_buffer *, std::set<uint64_t> >::iterator it=buffer_refs_.begin();
        it!=buffer_refs_.end();
        ++it) {
        std::set<uint64_t>& op_bases = it->second;
        ss << " " << it->first << " = ";
        for(std::set<uint64_t>::iterator oit=op_bases.begin();
            oit != op_bases.end();
            oit++) {
            ss << *oit << ", ";
        }
        ss << endl;
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
