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

#ifndef __BH_VE_UNI_BLOCK_HPP
#define __BH_VE_UNI_BLOCK_HPP

#include <set>
#include <vector>
#include <iostream>

#include <bh_instruction.hpp>

namespace bohrium {

class Block {
public:
    std::vector <Block> _block_list;
    bh_instruction *_instr = NULL;
    int rank;
    int64_t size;
    std::set<bh_instruction*> _sweeps;
    std::set<bh_base *> _news;
    std::set<bh_base *> _frees;
    std::set<bh_base *> _temps;
    bool _reshapable = false;

    // Returns true if this block is an instruction block, which has a
    // empty block list and a non-NULL instruction pointer
    bool isInstr() const {
        assert(_block_list.size() > 0 or _instr != NULL);
        return _block_list.size() == 0;
    }

    // Find the 'instr' in this block or in its children
    // Returns NULL if not found
    Block* findInstrBlock(const bh_instruction *instr);

    // Pretty print this block
    std::string pprint() const;

    // Return all instructions in the block (incl. nested blocks)
    void getAllInstr(std::vector<bh_instruction *> &out) const;
    std::vector<bh_instruction *> getAllInstr() const;

    // Returns true when all instructions within the block is system or if the block is empty()
    bool isSystemOnly() const {
        for (bh_instruction *instr: getAllInstr()) {
            if (not bh_opcode_is_system(instr->opcode)) {
                return false;
            }
        }
        return true;
    }

    // Validation check of this block
    bool validation() const;
};

// Merge the two blocks, 'a' and 'b', in that order. When 'based_on_block_b' is
// true, the meta-data such at size, rank etc. is taken from 'b' rather than 'a'
Block merge(const Block &a, const Block &b, bool based_on_block_b=false);

// Create a nested block based on 'instr_list' with the sets of new, free, and temp arrays given.
// The dimensions from zero to 'rank-1' are ignored.
// The 'size_of_rank_dim' specifies the size of the dimension 'rank'.
Block create_nested_block(std::vector<bh_instruction*> &instr_list, int rank, int64_t size_of_rank_dim,
                          const std::set<bh_instruction*> &news);

//Implements pprint of block
std::ostream& operator<<(std::ostream& out, const Block& b);


//Implements pprint of a vector of blocks
std::ostream& operator<<(std::ostream& out, const std::vector<Block>& b);

} // bohrium

#endif
