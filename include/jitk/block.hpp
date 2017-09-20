/*
This file is part of Bohrium and copyright (c) 2012 the Bohrium
team <http://www.bh107.org>.

Bohrium is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3
of the License, or (at your option) any later version.

Bohrium is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the
GNU Lesser General Public License along with Bohrium.

If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __BH_JITK_BLOCK_HPP
#define __BH_JITK_BLOCK_HPP

#include <set>
#include <vector>
#include <iostream>
#include <memory>
#include <boost/variant/variant.hpp>
#include <boost/variant/get.hpp>

#include <bh_instruction.hpp>

namespace bohrium {
namespace jitk {

/* Design Overview

   A block can represent two things in Bohrium:
    * A for-loop, i.e. a traversal of arrays over an axis/dimension (class LoopB)
    * A single instruction such as BH_ADD or BH_MULTIPLY (class InstrB)

*/

// Forward declaration
class Block;

// We use a shared pointer of an const instruction. The idea is to never change an instruction inplace
// instead, create a whole new instruction.
typedef std::shared_ptr<const bh_instruction> InstrPtr;

// Representation of a for-loop, which contains a list of nested loops (_block_list)
class LoopB {
public:
    // The rank of this loop block
    int rank;
    // List of sub-blocks
    std::vector <Block> _block_list;
    // Size of the loop
    int64_t size;
    // Sweep instructions within this loop
    std::set<InstrPtr> _sweeps;
    // New arrays within this loop
    std::set<bh_base *> _news;
    // Freed arrays within this loop
    std::set<bh_base *> _frees;
    // Is this loop and all its sub-blocks reshapable
    bool _reshapable = false;

    // Unique id of this block
    int _id;

    // Default Constructor
    LoopB() { static int id_count = 0; _id = id_count++; }

    // Search and replace 'subject' with 'replacement' and returns the number of hits
    int replaceInstr(InstrPtr subject, const bh_instruction &replacement);

    // Is this block innermost? (not counting instruction blocks)
    bool isInnermost() const;

    // Return all sub-blocks (incl. nested blocks)
    void getAllSubBlocks(std::vector<const LoopB *> &out) const;

    // Return all sub-blocks (excl. nested blocks)
    std::vector<const LoopB*> getLocalSubBlocks() const;

    // Return all instructions in the block (incl. nested blocks)
    void getAllInstr(std::vector<InstrPtr> &out) const;
    std::vector<InstrPtr> getAllInstr() const;

    // Return instructions in the block (excl. nested blocks)
    void getLocalInstr(std::vector<InstrPtr> &out) const;
    std::vector<InstrPtr> getLocalInstr() const;

    // Return all bases accessed by this block
    std::set<const bh_base*> getAllBases() const {
        std::set<const bh_base*> ret;
        for (InstrPtr instr: getAllInstr()) {
            std::set<const bh_base*> t = instr->get_bases_const();
            ret.insert(t.begin(), t.end());
        }
        return ret;
    }

    // Return all new arrays in this block (incl. nested blocks)
    void getAllNews(std::set<bh_base*> &out) const;
    std::set<bh_base*> getAllNews() const;

    // Return all freed arrays in this block (incl. nested blocks)
    void getAllFrees(std::set<bh_base*> &out) const;
    std::set<bh_base*> getAllFrees() const;

    // Return all local temporary arrays in this block (excl. nested blocks)
    // NB: The returned temporary arrays are the arrays that this block should declare
    void getLocalTemps(std::set<bh_base*> &out) const;
    std::set<bh_base*> getLocalTemps() const;

    // Return all temporary arrays in this block (incl. nested blocks)
    void getAllTemps(std::set<bh_base*> &out) const;
    std::set<bh_base*> getAllTemps() const;

    // Return all non-temporary arrays in this block (incl. nested blocks)
    void getAllNonTemps(std::set<bh_base*> &out) const;
    std::set<bh_base*> getAllNonTemps() const;

    // Returns true when all instructions within the block is system or if the block is empty()
    bool isSystemOnly() const;

    // Validation check of this block
    bool validation() const;

    // Finds the block and instruction that accesses 'base' last. If 'base' is NULL, any data access
    // is accepted thus the last instruction is returned.
    // Returns the block and the index of the instruction (or NULL and -1 if not accessed)
    std::pair<LoopB*, int64_t> findLastAccessBy(const bh_base *base);

    // Insert the system instruction 'instr' after the instruction that accesses 'base' last.
    // If no instruction accesses 'base' we insert it after the last instruction.
    // NB: Force reshape the instruction to match the instructions accesses 'base' last.
    void insert_system_after(InstrPtr instr, const bh_base *base);

    // Returns the amount of threading in this block (excl. nested blocks)
    uint64_t localThreading() const;

    // Pretty print this block
    std::string pprint(const char *newline="\n") const;

    // Updates the metadata such as the sets of sweeps, news, frees etc.
    void metadata_update();

    // Equality test based on the unique ID
    bool operator==(const LoopB &loop_block) const {
        return this->_id == loop_block._id;
    }
};

// Representation of a single instruction
class InstrB {
public:
    InstrPtr instr;
    int rank;
};

// A block can represent a for-loop (LoopB) or an instruction (InstrB)
class Block {
private:
    // This is the variant that contains either a LoopB or a InstrB.
    // Note, boost::blank is only used to catch programming errors
    boost::variant<boost::blank, LoopB, InstrB> _var;

public:

    // Default Constructor
    Block() {}

    // Loop Block Constructor
    explicit Block(const LoopB &loop_block) {
        assert(_var.which() == 0);
        _var = loop_block;
    }
    explicit Block(LoopB &&loop_block) {
        assert(_var.which() == 0);
        _var = loop_block;
    }

    // Instruction Block Constructor
    // Note, the rank is only to make pretty printing easier
    Block(const bh_instruction &instr, int rank) {
        assert(_var.which() == 0);
        InstrB _instr{std::make_shared<bh_instruction>(instr), rank};
        _var = std::move(_instr);
    }

    // Returns true if this block is an instruction block
    bool isInstr() const {
        return _var.which() == 2; // Notice, the third type in '_var' is 'InstrB'
    }

    // Retrieve the Loop Block
    LoopB &getLoop() {return boost::get<LoopB>(_var);}
    const LoopB &getLoop() const {return boost::get<LoopB>(_var);}

    // Retrieve the instruction within the instruction block
    const InstrPtr getInstr() const {return boost::get<InstrB>(_var).instr;}
    InstrPtr getInstr() {return boost::get<InstrB>(_var).instr;}
    void setInstr(const bh_instruction &instr) {
        assert(_var.which() == 0 or _var.which() == 2);
        boost::get<InstrB>(_var).instr.reset(new bh_instruction(instr));
    }

    // Return the rank of this block
    int rank() const {
        if (isInstr()) {
            return boost::get<InstrB>(_var).rank;
        } else {
            return getLoop().rank;
        }
    }

    // Validation check of this block
    bool validation() const;

    // Return all instructions in the block (incl. nested blocks)
    void getAllInstr(std::vector<InstrPtr> &out) const;
    std::vector<InstrPtr> getAllInstr() const;

    // Return all bases accessed by this block
    std::set<const bh_base*> getAllBases() const {
        std::set<const bh_base*> ret;
        for (InstrPtr instr: getAllInstr()) {
            std::set<const bh_base*> t = instr->get_bases_const();
            ret.insert(t.begin(), t.end());
        }
        return ret;
    }

    // Returns true when all instructions within this block is system or if the block is empty()
    bool isSystemOnly() const {
        if (isInstr()) {
            return bh_opcode_is_system(getInstr()->opcode);
        }
        for (const Block &b: getLoop()._block_list) {
            if (not b.isSystemOnly()) {
                return false;
            }
        }
        return true;
    }

    // Returns true when all instructions within this block is reshapable
    bool isReshapable() const {
        if (isInstr()) {
            return getInstr()->reshapable();
        } else {
            return getLoop()._reshapable;
        }
    }

    // Determines whether this block must be executed after 'other'
    bool depend_on(const Block &other) const {
        const std::vector<InstrPtr> this_instr_list = getAllInstr();
        const std::vector<InstrPtr> other_instr_list = other.getAllInstr();
        for (const InstrPtr this_instr: this_instr_list) {
            for (const InstrPtr other_instr: other_instr_list) {
                if (bh_instr_dependency(this_instr.get(), other_instr.get())) {
                    return true;
                }
            }
        }
        return false;
    }

    // Return number of bytes accessed in the computation
    uint64_t compute_size() const {
        uint64_t ret=0;
        for(const InstrPtr &instr: getAllInstr()) {
            for (const bh_view *view: instr->get_views()) {
                ret += bh_nelements(*view) * bh_type_size(view->base->type);
            }
        }
        return ret;
    }

    // Pretty print this block
    std::string pprint(const char *newline="\n") const;
};

// Merge the two loop blocks, 'a' and 'b', in that order.
LoopB merge(const LoopB &a, const LoopB &b);

// Create a nested block based on 'instr_list' fully (from 'rank' to the innermost block)
// NB: ALL instructions (excl. sysop) must be fusible and have the same dominating shape.
Block create_nested_block(const std::vector<InstrPtr> &instr_list, int rank = 0);

// Create a nested block based on 'instr_list' partially (only 'rank' dimension.
// The dimensions from zero to 'rank-1' are ignored.
// The 'size_of_rank_dim' specifies the size of the dimension 'rank'.
Block create_nested_block(const std::vector<InstrPtr> &instr_list, int rank, int64_t size_of_rank_dim);

// Returns the blocks that can be parallelized in 'block' (incl. 'block' and its sub-blocks)
// and the total amount of parallelism (in number of possible parallel threads)
std::pair<std::vector<const LoopB *>, uint64_t> util_find_threaded_blocks(const LoopB &block);

// Check if the two blocks 'b1' and 'b2' (in that order) are mergeable.
// 'avoid_rank0_sweep' will not allow fusion of sweeped and non-sweeped blocks at the root level
bool mergeable(const Block &b1, const Block &b2, bool avoid_rank0_sweep);

// Reshape and merges the two loop blocks 'l1' and 'l2' (in that order).
// NB: the loop blocks must be mergeable!
Block reshape_and_merge(const LoopB &l1, const LoopB &l2);

//Implements pprint of block
std::ostream& operator<<(std::ostream& out, const LoopB& b);

//Implements pprint of block
std::ostream& operator<<(std::ostream& out, const Block& b);

//Implements pprint of a vector of blocks
std::ostream& operator<<(std::ostream& out, const std::vector<Block>& b);

} // jit
} // bohrium


#endif
