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

namespace {

// Help function that swap the two axes, 'axis1' and 'axis2', in all instructions in 'instr_list'
vector<InstrPtr> swap_axis(const vector<InstrPtr> &instr_list, int64_t axis1, int64_t axis2) {
    vector<InstrPtr> ret;
    for (const InstrPtr &instr: instr_list) {
        bh_instruction tmp(*instr);
        tmp.transpose(axis1, axis2);
        ret.push_back(std::make_shared<bh_instruction>(tmp));
    }
    return ret;
}

// Help function that swap the 'parent' block with its 'child' block.
// NB: 'child' must point to a block in 'parent._block_list'
vector<Block> swap_blocks(const LoopB &parent, const LoopB *child) {
    vector<Block> ret;
    for (const Block &b: parent._block_list) {
        LoopB loop;
        loop.rank = parent.rank;
        if (b.isInstr() or &b.getLoop() != child) {
            loop.size = parent.size;
            loop._block_list.push_back(b);
            loop._frees.insert(parent._frees.begin(), parent._frees.end());
        } else {
            loop.size = child->size;
            const vector<InstrPtr> t = swap_axis(child->getAllInstr(), parent.rank, child->rank);
            loop._block_list.push_back(create_nested_block(t, child->rank, parent.size));
            loop._frees.insert(child->_frees.begin(), child->_frees.end());
        }
        loop.metadataUpdate();
        ret.push_back(Block(std::move(loop)));
    }
    return ret;
}

// Help function that find a loop block within 'parent' that it make sense to swappable
const LoopB *find_swappable_sub_block(const LoopB &parent) {
    // For each sweep, we look for a sub-block that contains that sweep instructions.
    for (const InstrPtr &sweep: parent._sweeps) {
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

// Help function that collapses 'axis' and 'axis+1' in all instructions within 'loop'
// Returns false if encountering a non-compatible instruction
bool collapse_instr_axes(LoopB &loop, const int axis) {
    for (Block &block: loop._block_list) {
        if (block.isInstr()) {
            bh_instruction instr(*block.getInstr());
            const int sa = instr.sweep_axis();
            assert(sa != axis);
            assert(sa != axis+1);
            if (sa == axis or sa == axis+1) {
                return false;
            }
            for (size_t i=0; i<instr.operand.size(); ++i) {
                bh_view &view = instr.operand[i];
                if (not bh_is_constant(&view)) {
                    int _axis = axis;
                    if (i==0 and bh_opcode_is_reduction(instr.opcode)) {
                        _axis = sa < _axis ? _axis-1 : _axis;
                    }
                    assert(view.ndim > _axis+1);
                    if (view.shape[_axis+1] * view.stride[_axis+1] != view.stride[_axis]) {
                        return false;
                    }
                    view.shape[_axis] *= view.shape[_axis+1];
                    view.stride[_axis] = view.stride[_axis+1];
                }
            }
            instr.remove_axis(axis+1);
            block.setInstr(instr);
        } else {
            --block.getLoop().rank;
            if (not collapse_instr_axes(block.getLoop(), axis)) {
                return false;
            }
        }
    }
    loop.metadataUpdate();
    assert(loop.validation());
    return true;
}

// Help function that collapses 'loop' with its child if possible
bool collapse_loop_with_child(LoopB &loop) {
    // In order to be collapsable, 'loop' can only have one child, that child must be a loop, and both 'loop'
    // and the child cannot be sweeped
    if (loop._sweeps.empty() and loop._block_list.size() == 1 and loop._reshapable) {
        Block &child = loop._block_list[0];
        if ((not child.isInstr()) and child.getLoop()._sweeps.empty() and child.getLoop()._reshapable) {
            // Let's collapse with our single child
            loop.size *= child.getLoop().size;
            loop._frees.insert(child.getLoop()._frees.begin(), child.getLoop()._frees.end());
            // NB: we need the temporary step in order to avoid copying deleted data
            auto tmp = std::move(child.getLoop()._block_list);
            loop._block_list = std::move(tmp);
            return collapse_instr_axes(loop, loop.rank);
        }
    }
    return false;
}
}

void push_reductions_inwards(vector<Block> &block_list) {
    vector<Block> block_list2(block_list);
    for(Block &b: block_list2) {
        if (not b.isInstr()) {
            push_reductions_inwards(b.getLoop()._block_list);
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
    block_list = ret;
}

void split_for_threading(vector<Block> &block_list, uint64_t min_threading) {
    vector<Block> ret;

    for (const Block &block: block_list) {
        // For now, we cannot make an instruction or a sweeped block threadable
        if (block.isInstr() or block.getLoop()._sweeps.size() > 0) {
            ret.push_back(block);
            continue;
        }
        const LoopB &loop = block.getLoop();
        uint64_t max_nelem = 0; // The maximum number of element in loop, which tells use the best-case scenario
        for (const InstrPtr &instr: loop.getAllInstr()) {
            if (instr->operand.size() > 0) {
                const uint64_t nelem = static_cast<uint64_t>(bh_nelements(instr->operand[0]));
                if (nelem > max_nelem)
                    max_nelem = nelem;
            }
        }
        if (loop._block_list.size() > 1 // We need minimum two blocks (instructions) in order to split!
            and max_nelem > min_threading // Is it even possible to achieve our goal?
            and parallel_ranks(loop).second < min_threading) { // Is the goal already achieved?

            for (auto it = loop._block_list.begin(); it != loop._block_list.end(); ++it) {
                // First we will place all sub-blocks that cannot be threaded in a shared block
                {
                    LoopB newloop{loop.rank, loop.size};
                    while (it != loop._block_list.end() and (it->isInstr() or it->getLoop()._sweeps.size() > 0)) {
                        assert(it->rank() == newloop.rank+1);
                        newloop._block_list.push_back(*it);
                        ++it;
                    }
                    if (not newloop._block_list.empty()) {
                        newloop.metadataUpdate();
                        ret.push_back(Block(std::move(newloop)));
                    }
                }
                // Then we place the highly threaded sub-block in its own block
                if (it != loop._block_list.end()) {
                    assert(not it->isInstr());
                    assert(it->getLoop()._sweeps.size() == 0);
                    LoopB newloop{loop.rank, loop.size};
                    newloop._block_list.push_back(*it);
                    newloop.metadataUpdate();
                    ret.push_back(Block(std::move(newloop)));
                }
                // TODO: for now the last new block gets all frees.
                assert(not ret.back().isInstr());
                ret.back().getLoop()._frees = loop._frees;

                // Make sure we don't overflow
                if (it == loop._block_list.end()) {
                    break;
                }
            }
        } else {
            ret.push_back(block);
        }
    }
    block_list = ret;
}

void collapse_redundant_axes(vector<Block> &block_list) {
    for(Block &b: block_list) {
        if (not b.isInstr()) {
            collapse_redundant_axes(b.getLoop()._block_list);
        }
    }
    vector<Block> ret;
    for (const Block &block: block_list) {
        if (not block.isInstr()) {
            Block b(block);
            if (collapse_loop_with_child(b.getLoop())) {
                ret.push_back(std::move(b));
            } else { // If it didn't succeeded, we just push the original block
                ret.push_back(block);
            }
        } else {
            ret.push_back(block);
        }
    }
    block_list = ret;
}
} // jitk
} // bohrium

