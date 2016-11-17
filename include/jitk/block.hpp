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

#include <bh_instruction.hpp>

namespace bohrium {
namespace jitk {

class Block;

class LoopB {
public:
    int rank;
    std::vector <Block> _block_list;
    int64_t size;
    std::set<bh_instruction*> _sweeps;
    std::set<bh_base *> _news;
    std::set<bh_base *> _frees;
    bool _reshapable = false;

    // Unique id of this block
    int _id;

    // Default Constructor
    LoopB() { static int id_count = 0; _id = id_count++; }

    // Find the 'instr' in this block or in its children
    // Returns NULL if not found
    Block* findInstrBlock(const bh_instruction *instr);

    // Is this block innermost? (not counting instruction blocks)
    bool isInnermost() const;

    // Return all sub-blocks (incl. nested blocks)
    void getAllSubBlocks(std::vector<const LoopB *> &out) const;

    // Return all sub-blocks (excl. nested blocks)
    std::vector<const Block*> getLocalSubBlocks() const;

    // Return all instructions in the block (incl. nested blocks)
    void getAllInstr(std::vector<bh_instruction*> &out) const;
    std::vector<bh_instruction*> getAllInstr() const;

    // Return instructions in the block (excl. nested blocks)
    void getLocalInstr(std::vector<bh_instruction*> &out) const;
    std::vector<bh_instruction*> getLocalInstr() const;

    // Return all bases accessed by this block
    std::set<bh_base*> getAllBases() const {
        std::set<bh_base*> ret;
        for (bh_instruction *instr: getAllInstr()) {
            std::set<bh_base*> t = instr->get_bases();
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

    // Returns true when all instructions within the block is system or if the block is empty()
    bool isSystemOnly() const;

    // Validation check of this block
    bool validation() const;

    // Determines whether this block must be executed after 'other'
    bool depend_on(const Block &other) const;

    // Finds the block and instruction that accesses 'base' last. If 'base' is NULL, any data access
    // is accepted thus the last instruction is returned.
    // Returns the block and the index of the instruction (or NULL and -1 if not accessed)
    std::pair<LoopB*, int64_t> findLastAccessBy(const bh_base *base);

    // Insert the system instruction 'instr' after the instruction that accesses 'base' last.
    // If no instruction accesses 'base' we insert it after the last instruction.
    // NB: Force reshape the instruction to match the instructions accesses 'base' last.
    void insert_system_after(bh_instruction *instr, const bh_base *base);

    // Pretty print this block
    std::string pprint(const char *newline="\n") const;

    // Equality test based on the unique ID
    bool operator==(const LoopB &loop_block) const {
        return this->_id == loop_block._id;
    }
};


class Block {
    LoopB loop;
public:

    bh_instruction *_instr = NULL;
    int _rank=-1;

    // Default Constructor
    Block() {}

    // Loop Block Constructor
    explicit Block(const LoopB &loop_block) {
        loop = loop_block;
    }
    explicit Block(LoopB &&loop_block) {
        loop = loop_block;
    }

    // Instruction Block  Constructor
    // Note, the rank is only to make pretty printing easier
    Block(bh_instruction *instr, int rank) {
        _instr = instr;
        this->_rank = rank;
        assert(rank < 20);
    }

    // Returns true if this block is an instruction block, which has a
    // empty block list and a non-NULL instruction pointer
    bool isInstr() const {
        return loop._block_list.size() == 0;
    }

    LoopB &getLoop() {return loop;}
    const LoopB &getLoop() const {return loop;}

    int rank() const {
        if (isInstr()) {
            return _rank;
        } else {
            return loop.rank;
        }
    }

    // Pretty print this block
    std::string pprint(const char *newline="\n") const;

    // Return all instructions in the block (incl. nested blocks)
    void getAllInstr(std::vector<bh_instruction*> &out) const;
    std::vector<bh_instruction*> getAllInstr() const;

    // Return all bases accessed by this block
    std::set<bh_base*> getAllBases() const {
        std::set<bh_base*> ret;
        for (bh_instruction *instr: getAllInstr()) {
            std::set<bh_base*> t = instr->get_bases();
            ret.insert(t.begin(), t.end());
        }
        return ret;
    }

    // Returns true when all instructions within the block is system or if the block is empty()
    bool isSystemOnly() const {
        if (isInstr()) {
            return bh_opcode_is_system(_instr->opcode);
        }
        for (const Block &b: loop._block_list) {
            if (not b.isSystemOnly()) {
                return false;
            }
        }
        return true;
    }

    bool isReshapable() const {
        if (isInstr()) {
            assert(_instr != NULL);
            return _instr->reshapable();
        } else {
            return loop._reshapable;
        }
    }

    // Validation check of this block
    bool validation() const;

    // Determines whether this block must be executed after 'other'
    bool depend_on(const Block &other) const {
        const std::vector<bh_instruction*> this_instr_list = getAllInstr();
        const std::vector<bh_instruction*> other_instr_list = other.getAllInstr();
        for (const bh_instruction *this_instr: this_instr_list) {
            for (const bh_instruction *other_instr: other_instr_list) {
                if (this_instr != other_instr and
                    bh_instr_dependency(this_instr, other_instr)) {
                    return true;
                }
            }
        }
        return false;
    }
};

// Merge the two loop blocks, 'a' and 'b', in that order.
LoopB merge(const LoopB &a, const LoopB &b);

// Create a nested block based on 'instr_list' with the sets of new, free, and temp arrays given.
// The dimensions from zero to 'rank-1' are ignored.
// The 'size_of_rank_dim' specifies the size of the dimension 'rank'.
Block create_nested_block(std::vector<bh_instruction *> &instr_list, int rank, int64_t size_of_rank_dim);

// Returns the blocks that can be parallelized in 'block' (incl. 'block' and its sub-blocks)
// and the total amount of parallelism (in number of possible parallel threads)
std::pair<std::vector<const LoopB *>, uint64_t> find_threaded_blocks(const LoopB &block);

// Check if it is possible to merge 'a' and 'a' (in that order)
bool merge_possible(const Block &a, const Block &b);

// Merges the two blocks 'a' and 'a' (in that order) if they are fusible.
// NB: 'a' or 'b' might be reshaped in order to make the merge legal
// Returns the new block and a flag indicating whether the merge was performed
std::pair<Block, bool> merge_if_possible(Block &a, Block &b);

//Implements pprint of block
std::ostream& operator<<(std::ostream& out, const LoopB& b);

//Implements pprint of block
std::ostream& operator<<(std::ostream& out, const Block& b);

//Implements pprint of a vector of blocks
std::ostream& operator<<(std::ostream& out, const std::vector<Block>& b);

} // jit
} // bohrium


#endif
