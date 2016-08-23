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

    // Returns true if this block is an instruction block
    // NB: Even an instruction block can have '_instr == NULL'
    bool isInstr() const {
        return _block_list.size() == 0;
    }

    // Finds the total size of this block and its children
    int64_t totalSize() const {
        if (isInstr())
            return 0;
        int64_t sum = 0;
        for (const Block &b: _block_list) {
            sum += b.totalSize();
        }
        return sum * size;
    }

    // Find the 'instr' in this block or in its children
    // Returns NULL if not found
    Block* findInstrBlock(const bh_instruction *instr);

    // Pretty print this block
    std::string pprint() const;

    // Return all instructions in the block (incl. nested blocks)
    void getAllInstr(std::vector<bh_instruction *> &out) const;
    std::vector<bh_instruction *> getAllInstr() const;

    // Merge this block with 'block' (in that order)
    void merge(const Block &block);
};

Block create_nested_block(std::vector<bh_instruction>::iterator begin, std::vector<bh_instruction>::iterator end, int rank, std::set<bh_base *> &news, std::set<bh_base *> &frees, std::set<bh_base *> &temps, bool reshapable);

//Implements pprint of block
std::ostream& operator<<(std::ostream& out, const Block& b);


//Implements pprint of a vector of blocks
std::ostream& operator<<(std::ostream& out, const std::vector<Block>& b);

} // bohrium

#endif
