#include <iomanip>
#include "block.hpp"

using namespace std;

namespace kp{
namespace core{

const char Block::TAG[] = "Block";

Block::Block(SymbolTable& globals, Program& tac_program)
: block_(), globals_(globals), tac_program_(tac_program), symbol_text_(""), symbol_("")
{
    const uint64_t buffer_capacity = globals_.capacity()+1;
    const uint64_t tac_capacity = tac_program_.capacity()+1;

    block_.buffers = new kp_buffer*[buffer_capacity];
    block_.operands = new kp_operand*[buffer_capacity];
    block_.tacs = new int64_t[tac_capacity];
    block_.array_tacs = new int64_t[tac_capacity];
}

Block::~Block()
{
    if (block_.buffers) {                       // Buffers
        delete[] block_.buffers;
        block_.buffers = NULL;
    }
    
    if (block_.operands) {                      // Operands
        delete[] block_.operands;
        block_.operands = NULL;
    }

    if (block_.tacs) {                          // Block tacs
        delete[] block_.tacs;
        block_.tacs = NULL;
    }

    if (block_.array_tacs) {                    // Block array tacs
        delete[] block_.array_tacs;
        block_.array_tacs = NULL;
    }
	clear();
}

void Block::clear(void)
{                                   // Reset the state of the kp_block (C interface)
    block_.nbuffers = 0;            // Buffers
    block_.noperands = 0;           // Operands

    block_.iterspace.layout = KP_SCALAR_TEMP;   // Iteraton space
    block_.iterspace.ndim = 0;
    block_.iterspace.shape = NULL;
    block_.iterspace.nelem = 0;

    block_.omask = 0;               // Operation mask
    block_.ntacs = 0;               // Block tacs
    block_.narray_tacs = 0;         // Block array tacs
                                    // End of kp_block reset, dynamic memory is reused.

    buffer_ids_.clear();        // Reset the state of Block (C++ interface)
    buffer_refs_.clear();

    global_to_local_.clear();   // global to local kp_operand mapping
    local_to_global_.clear();   // local to global kp_operand mapping

    symbol_text_    = "";       // textual block-symbol representation
    symbol_         = "";       // hashed block-symbol representation
}

void Block::_compose(bh_ir_kernel& krnl, bool array_contraction, size_t prg_idx)
{
    kp_tac& tac = tac_program_[prg_idx];

    block_.tacs[block_.ntacs++] = prg_idx;  // <-- All tacs
    block_.omask |= tac.op;                 // Update omask

    if ((tac.op & (KP_ARRAY_OPS))>0) {      // <-- Only array operations
        block_.array_tacs[block_.narray_tacs++] = prg_idx;
    }

    switch(tac_noperands(tac)) {
        case 3:
            _localize_scope(tac.in2);                       // Localize scope,      in2
            if ((tac.op & (KP_ARRAY_OPS))>0) {
                if (array_contraction && (krnl.get_temps().find((bh_base*)globals_[tac.in2].base)!=krnl.get_temps().end())) {
                    globals_.turn_contractable(tac.in2);    // Mark contractable,   in2
                }
                _bufferize(tac.in2);                        // Note down buffer id, in2
            }
        case 2:
            _localize_scope(tac.in1);                       // Localize scope,      in1
            if ((tac.op & (KP_ARRAY_OPS))>0) {
                if (array_contraction && (krnl.get_temps().find((bh_base*)globals_[tac.in1].base)!=krnl.get_temps().end())) {
                    globals_.turn_contractable(tac.in1);    // Mark contractable,   in1
                }
                _bufferize(tac.in1);                        // Note down buffer id, in1
            }
        case 1:
            _localize_scope(tac.out);                       // Localize scope,      out
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
    for(std::vector<uint64_t>::iterator idx_it = krnl.instr_indexes.begin();
        idx_it != krnl.instr_indexes.end();
        ++idx_it) {

        _compose(krnl, array_contraction, *idx_it);
    }

    if (array_contraction) {    // Turn kernel-temps into scalars aka array-contraction
        for (bh_base* base: krnl.get_temps()) {
            for(int64_t oidx = 0; oidx < noperands();  ++oidx) {
                if (operand(oidx).base == (kp_buffer*)base) {
                    globals_.turn_contractable(local_to_global(oidx));
                }
            }
        }
    }

    _update_iterspace();        // Update the iteration space
}

void Block::compose(bh_ir_kernel& krnl, size_t prg_idx)
{
    _compose(krnl, false, prg_idx);
    _update_iterspace();        // Update the iteration space
}

void Block::_bufferize(size_t global_idx)
{
    // Maintain references to buffers within the block.
    if ((globals_[global_idx].layout & (KP_DYNALLOC_LAYOUT))>0) {
        kp_buffer* buffer = globals_[global_idx].base;

        std::map<kp_buffer*, size_t>::iterator buf = buffer_ids_.find(buffer);
        if (buf == buffer_ids_.end()) {
            size_t buffer_id = block_.nbuffers++;
            buffer_ids_.insert(pair<kp_buffer *, size_t>(
                buffer,
                buffer_id
            ));
            block_.buffers[buffer_id] = buffer;
        }

        buffer_refs_[buffer].insert(global_idx);
    }
} 

size_t Block::_localize_scope(size_t global_idx)
{
    //
    // Reuse kp_operand identifiers: Detect if we have seen it before and reuse the index.
    int64_t local_idx = 0;
    bool found = false;
    for(int64_t i=0; i<block_.noperands; ++i) {
        if (!core::equivalent(*block_.operands[i], globals_[global_idx])) {
            continue; // Not equivalent, continue search.
        }
        // Found one! Use it instead of the incremented identifier.
        local_idx = i;
        found = true;
        break;
    }

    // Create the kp_operand in block-scope
    if (!found) {
        local_idx = block_.noperands;
        block_.operands[local_idx] = &globals_[global_idx];
        ++block_.noperands;
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
    for(int64_t oidx=0; oidx <block_.noperands; ++oidx) {
        operands_ss << "~" << oidx;
        operands_ss << core::layout_text_shand(block_.operands[oidx]->layout);
        operands_ss << core::etype_text_shand(block_.operands[oidx]->etype);

        // Let the "Restrictable" flag be part of the symbol.
        if (buffer_refs_[block_.operands[oidx]->base].size()==1) {
            operands_ss << "R";
        } else {
            operands_ss << "A";
        }
    }

    //
    // Program
    bool first = true;
    for(int64_t tac_iter=0; tac_iter<block_.ntacs; ++tac_iter) {
        kp_tac& tac = this->tac(tac_iter);
       
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
        switch(tac_noperands(tac)) {
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

kp_buffer& Block::buffer(size_t buffer_id)
{
    return *block_.buffers[buffer_id];
}

size_t Block::resolve_buffer(kp_buffer* buffer)
{
    std::map<kp_buffer*, size_t>::iterator buf = buffer_ids_.find(buffer);
    if (buf == buffer_ids_.end()) {
        // TODO: Raise exception
    }
    return buf->second;
}

kp_buffer** Block::buffers(void)
{
    return block_.buffers;
}

int64_t Block::nbuffers()
{
    return block_.nbuffers;
}

size_t Block::buffer_refcount(kp_buffer* buffer)
{
    return buffer_refs_[buffer].size();
}

kp_operand & Block::operand(size_t local_idx)
{
    return *block_.operands[local_idx];
}

kp_operand** Block::operands(void)
{
    return block_.operands;
}

int64_t Block::noperands(void) const
{
    return block_.noperands;
}

size_t Block::global_to_local(size_t global_idx) const
{
    return global_to_local_.find(global_idx)->second;
}

size_t Block::local_to_global(size_t local_idx) const
{
    return local_to_global_.find(local_idx)->second;
}

kp_tac& Block::tac(size_t idx) const
{
    return tac_program_[block_.tacs[idx]];
}

kp_tac & Block::array_tac(size_t idx) const
{
    return tac_program_[block_.array_tacs[idx]];
}

uint32_t Block::omask(void)
{
    return block_.omask;
}

size_t Block::ntacs(void) const
{
    return block_.ntacs;
}

size_t Block::narray_tacs(void) const
{
    return block_.narray_tacs;
}

string Block::symbol(void) const
{
    return symbol_;
}

string Block::symbol_text(void) const
{
    return symbol_text_;
}

kp_iterspace& Block::iterspace(void)
{
    return block_.iterspace;
}

void Block::_update_iterspace(void)
{       
    //
    // Determine layout, ndim and shape
    for(size_t tac_idx=0; tac_idx<ntacs(); ++tac_idx) {
        kp_tac & tac = this->tac(tac_idx);
        if (not ((tac.op & (KP_ARRAY_OPS))>0)) {   // Only interested in array ops
            continue;
        }
        if ((tac.op & (KP_REDUCE_COMPLETE | KP_REDUCE_PARTIAL))>0) {    // Reductions are weird
            if (globals_[tac.in1].layout >= block_.iterspace.layout) {  // Iterspace
                block_.iterspace.layout = globals_[tac.in1].layout;
                block_.iterspace.ndim  = globals_[tac.in1].ndim;
                block_.iterspace.shape = globals_[tac.in1].shape;
            }
            if (globals_[tac.out].layout > block_.iterspace.layout) {
                block_.iterspace.layout = globals_[tac.out].layout;
            }
        } else {
            switch(tac_noperands(tac)) {
                case 3:
                    if (globals_[tac.in2].layout > block_.iterspace.layout) {   // Iterspace
                        block_.iterspace.layout = globals_[tac.in2].layout;
                        if (block_.iterspace.layout > KP_SCALAR_TEMP) {
                            block_.iterspace.ndim  = globals_[tac.in2].ndim;
                            block_.iterspace.shape = globals_[tac.in2].shape;
                        }
                    }
                case 2:
                    if (globals_[tac.in1].layout > block_.iterspace.layout) {   // Iterspace
                        block_.iterspace.layout = globals_[tac.in1].layout;
                        if (block_.iterspace.layout > KP_SCALAR_TEMP) {
                            block_.iterspace.ndim  = globals_[tac.in1].ndim;
                            block_.iterspace.shape = globals_[tac.in1].shape;
                        }
                    }
                case 1:
                    if (globals_[tac.out].layout > block_.iterspace.layout) {   // Iterspace
                        block_.iterspace.layout = globals_[tac.out].layout;
                        if (block_.iterspace.layout > KP_SCALAR_TEMP) {
                            block_.iterspace.ndim  = globals_[tac.out].ndim;
                            block_.iterspace.shape = globals_[tac.out].shape;
                        }
                    }
                default:
                    break;
            }
        }
    }

    if (NULL != block_.iterspace.shape) {               // Determine number of elements
        block_.iterspace.nelem = 1;
        for(int k=0; k<block_.iterspace.ndim; ++k) {
            block_.iterspace.nelem *= block_.iterspace.shape[k];
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
    ss << "BLOCK [" << symbol() << endl;
    ss << "TACS " << "omask(" << omask() << "), ntacs(" << ntacs() << ") {" << endl;
    for(uint64_t tac_idx=0; tac_idx<ntacs(); ++tac_idx) {
        ss << " " << tac_text(tac(tac_idx)) << endl;
    }
    ss << "}" << endl;

    ss << "OPERANDS (" << noperands() << ") {" << endl;
    for(int64_t opr_idx=0; opr_idx < noperands(); ++opr_idx) {
        kp_operand & opr = operand(opr_idx);
        ss << " loc_idx(" << opr_idx << ")";
        ss << " gbl_idx(" << local_to_global(opr_idx) << ") = ";
        ss << operand_text(opr);
        ss << endl;
    }
    ss << "}" << endl;

    ss << "BUFFER_REFS {" << endl;
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
    ss << " LAYOUT = " << layout_text(block_.iterspace.layout) << "," << endl;
    ss << " NDIM   = " << block_.iterspace.ndim << "," << endl;
    ss << " SHAPE  = {"; 
    for(int64_t dim=0; dim < block_.iterspace.ndim; ++dim) {
        ss << block_.iterspace.shape[dim];
        if (dim != (block_.iterspace.ndim-1)) {
            ss << ", ";
        }
    }
    ss << "}" << endl;
    ss << " NELEM  = " << block_.iterspace.nelem << endl;
    ss << "}" << endl;
    ss << "]" << endl;
    return ss.str();
}

kp_block& Block::meta(void)
{
    return block_;
}

}}
