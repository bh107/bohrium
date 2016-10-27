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

#include <jitk/fuser.hpp>

using namespace std;

namespace bohrium {
namespace jitk {

// Unnamed namespace for all fuser help functions
namespace {

// Check if 'a' and 'b' supports data-parallelism when merged
bool data_parallel_compatible(const bh_instruction *a, const bh_instruction *b)
{
    if(bh_opcode_is_system(a->opcode) || bh_opcode_is_system(b->opcode))
        return true;

    const int a_nop = bh_noperands(a->opcode);
    for(int i=0; i<a_nop; ++i)
    {
        if(not bh_view_disjoint(&b->operand[0], &a->operand[i])
           && not bh_view_aligned(&b->operand[0], &a->operand[i]))
            return false;
    }
    const int b_nop = bh_noperands(b->opcode);
    for(int i=0; i<b_nop; ++i)
    {
        if(not bh_view_disjoint(&a->operand[0], &b->operand[i])
           && not bh_view_aligned(&a->operand[0], &b->operand[i]))
            return false;
    }
    return true;
}

// Check if 'b1' and 'b2' supports data-parallelism when merged
bool data_parallel_compatible(const Block &b1, const Block &b2) {
    for (const bh_instruction *i1 : b1.getAllInstr()) {
        for (const bh_instruction *i2 : b2.getAllInstr()) {
            if (not data_parallel_compatible(i1, i2))
                return false;
        }
    }
    return true;
}

// Check if 'block' accesses the output of a sweep in 'sweeps'
bool sweeps_accessed_by_block(const set<bh_instruction*> &sweeps, const Block &block) {
    for (bh_instruction *instr: sweeps) {
        assert(bh_noperands(instr->opcode) > 0);
        auto bases = block.getAllBases();
        if (bases.find(instr->operand[0].base) != bases.end())
            return true;
    }
    return false;
}
} // Unnamed namespace


vector<Block> fuser_singleton(vector<bh_instruction> &instr_list, const set<bh_instruction*> &news) {

    // Creates the _block_list based on the instr_list
    vector<Block> block_list;
    for (auto instr=instr_list.begin(); instr != instr_list.end(); ++instr) {
        int nop = bh_noperands(instr->opcode);
        if (nop == 0)
            continue; // Ignore noop instructions such as BH_NONE or BH_TALLY

        // Let's try to simplify the shape of the instruction
        if (instr->reshapable()) {
            const vector<int64_t> dominating_shape = instr->dominating_shape();
            assert(dominating_shape.size() > 0);

            const int64_t totalsize = std::accumulate(dominating_shape.begin(), dominating_shape.end(), 1, \
                                                      std::multiplies<int64_t>());
            const vector<int64_t> shape = {totalsize};
            instr->reshape(shape);
        }
        // Let's create the block
        const vector<int64_t> dominating_shape = instr->dominating_shape();
        assert(dominating_shape.size() > 0);
        int64_t size_of_rank_dim = dominating_shape[0];
        vector<bh_instruction*> single_instr = {&instr[0]};
        block_list.push_back(create_nested_block(single_instr, 0, size_of_rank_dim, news));
    }
    return block_list;
}


vector<Block> fuser_serial(vector<Block> &block_list, const set<bh_instruction*> &news) {
    vector<Block> ret;
    for (auto it = block_list.begin(); it != block_list.end(); ) {
        ret.push_back(*it);
        Block &cur = ret.back();
        ++it;
        if (cur.isInstr()) {
            continue; // We should never fuse instruction blocks
        }
        // Let's search for fusible blocks
        for (; it != block_list.end(); ++it) {
            if (it->isInstr())
                break;
            if (not data_parallel_compatible(cur, *it))
                break;
            if (sweeps_accessed_by_block(cur._sweeps, *it))
                break;
            assert(cur.rank == it->rank);

            // Check for perfect match, which is directly mergeable
            if (cur.size == it->size) {
                cur = merge(cur, *it);
                continue;
            }
            // Check fusibility of reshapable blocks
            if (it->_reshapable && it->size % cur.size == 0) {
                vector<bh_instruction *> cur_instr = cur.getAllInstr();
                vector<bh_instruction *> it_instr = it->getAllInstr();
                cur_instr.insert(cur_instr.end(), it_instr.begin(), it_instr.end());
                Block b = create_nested_block(cur_instr, it->rank, cur.size, news);
                assert(b.size == cur.size);
                cur = b;
                continue;
            }
            if (cur._reshapable && cur.size % it->size == 0) {
                vector<bh_instruction *> cur_instr = cur.getAllInstr();
                vector<bh_instruction *> it_instr = it->getAllInstr();
                cur_instr.insert(cur_instr.end(), it_instr.begin(), it_instr.end());
                Block b = create_nested_block(cur_instr, cur.rank, it->size, news);
                assert(b.size == it->size);
                cur = b;
                continue;
            }

            // We couldn't find any shape match
            break;
        }
        // Let's fuse at the next rank level
        cur._block_list = fuser_serial(cur._block_list, news);
    }
    return ret;
}


} // jitk
} // bohrium
