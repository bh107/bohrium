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

#include <jitk/transformer.hpp>

using namespace std;

namespace bohrium {
namespace jitk {

vector<InstrPtr> swap_axis(const vector<InstrPtr> &instr_list, int64_t axis1, int64_t axis2) {
    vector<InstrPtr> ret;
    for (const InstrPtr &instr: instr_list) {
        bh_instruction tmp(*instr);
        tmp.transpose(axis1, axis2);
        ret.push_back(std::make_shared<bh_instruction>(tmp));
    }
    return ret;
}

vector<Block> swap_blocks(const LoopB &parent, const LoopB *child) {
    vector<Block> ret;
    for (const Block &b: parent._block_list) {
        LoopB loop;
        loop.rank = parent.rank;
        if (b.isInstr() or &b.getLoop() != child) {
            loop.size = parent.size;
            loop._block_list.push_back(b);
        } else {
            loop.size = child->size;
            const vector<InstrPtr> t = swap_axis(child->getAllInstr(), parent.rank, child->rank);
            loop._block_list.push_back(create_nested_block(t, child->rank, parent.size));
        }
        loop.metadata_update();
        ret.push_back(Block(std::move(loop)));
    }
    return ret;
}

const LoopB *find_swappable_sub_block(const LoopB &parent) {
    // For each sweep, we look for a sub-block that contains that sweep instructions.
    for (const InstrPtr sweep: parent._sweeps) {
        for (const Block &b: parent._block_list) {
            if (not b.isInstr()) {
                for (const Block &instr_block: b.getLoop()._block_list) {
                    if (instr_block.isInstr()) {
                        if (*instr_block.getInstr() == *sweep) {
                            return &b.getLoop();
                        }
                    }
                }
            }
        }
    }
    return NULL;
}

vector<Block> push_reductions_inwards(const vector<Block> &block_list) {
    vector<Block> block_list2(block_list);
    for(Block &b: block_list2) {
        if (not b.isInstr()) {
            b.getLoop()._block_list = push_reductions_inwards(b.getLoop()._block_list);
        }
    }
    vector<Block> ret;
    for(const Block &b: block_list2) {
        const LoopB *swappable;
        if (not b.isInstr() and (swappable = find_swappable_sub_block(b.getLoop())) != NULL)  {
            const vector<Block>tmp = swap_blocks(b.getLoop(), swappable);
            ret.insert(ret.end(), tmp.begin(), tmp.end());
        } else {
            ret.push_back(b);
        }
    }
    return ret;
}

vector<Block> split_for_threading(const vector<Block> &block_list, uint64_t min_threading, uint64_t cur_threading) {
    vector<Block> ret;

    for (const Block &block: block_list) {
        // For now, we cannot make an instruction or a sweeped block threadable
        if (block.isInstr() or block.getLoop()._sweeps.size() > 0) {
            ret.push_back(block);
            continue;
        }
        const LoopB &loop = block.getLoop();
        uint64_t max_nelem = 0; // The maximum number of element in loop, which tells use the best-case scenario
        for (const InstrPtr instr: loop.getAllInstr()) {
            if (bh_noperands(instr->opcode) > 0) {
                const uint64_t nelem = static_cast<uint64_t>(bh_nelements(instr->operand[0]));
                if (nelem > max_nelem)
                    max_nelem = nelem;
            }
        }
        if (loop._block_list.size() > 1 // We need minimum two blocks in order to split!
            and max_nelem > min_threading // Is it even possible to achieve our goal?
            and find_threaded_blocks(loop).second < min_threading-cur_threading) { // Is the goal already achieved?

            for (auto it = loop._block_list.begin(); it != loop._block_list.end(); ++it) {
                // First we will place all sub-blocks that cannot be threaded in a shared block
                {
                    LoopB newloop;
                    newloop.rank = loop.rank;
                    newloop.size = loop.size;
                    while (it != loop._block_list.end() and (it->isInstr() or it->getLoop()._sweeps.size() > 0)) {
                        assert(it->rank() == newloop.rank+1);
                        newloop._block_list.push_back(*it);
                        ++it;
                    }
                    newloop.metadata_update();
                    if (not newloop._block_list.empty()) {
                        ret.push_back(Block(std::move(newloop)));
                    }
                }
                // Then we place the highly threaded sub-block in its own block
                if (it != loop._block_list.end()) {
                    assert(not it->isInstr());
                    assert(it->getLoop()._sweeps.size() == 0);
                    LoopB newloop;
                    newloop.rank = loop.rank;
                    newloop.size = loop.size;
                    newloop._block_list.push_back(*it);
                    newloop.metadata_update();
                    ret.push_back(Block(std::move(newloop)));
                } else {
                    break;
                }
            }
        } else {
            ret.push_back(block);
        }
    }
    return ret;
}

} // jitk
} // bohrium

