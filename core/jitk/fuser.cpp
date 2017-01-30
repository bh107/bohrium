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

#include <fstream>
#include <numeric>
#include <queue>
#include <cassert>

#include <jitk/fuser.hpp>
#include <jitk/graph.hpp>

using namespace std;

namespace bohrium {
namespace jitk {

void simplify_instr(bh_instruction &instr) {
    if (bh_noperands(instr.opcode) == 0) {
        return;
    }

    // Let's start by removing redundant 1-sized dimensions (but make sure we don't remove all dimensions!)
    {
        const vector<int64_t> dominating_shape = instr.dominating_shape();
        const int sa = instr.sweep_axis();
        size_t ndim_left = bh_opcode_is_reduction(instr.opcode)?dominating_shape.size()-1:dominating_shape.size();
        for (int64_t i=dominating_shape.size()-1; i >= 0 and ndim_left > 1; --i) {
            if (sa != i and dominating_shape[i] == 1) {
                instr.remove_axis(i);
                --ndim_left;
            }
        }
    }

    // Let's try to simplify the shape of the instruction
    if (instr.max_ndim() >  1 and instr.reshapable()) {
        const vector<int64_t> dominating_shape = instr.dominating_shape();
        assert(dominating_shape.size() > 0);

        const int64_t totalsize = std::accumulate(dominating_shape.begin(), dominating_shape.end(), int64_t{1}, \
                                                      std::multiplies<int64_t>());
        const vector<int64_t> shape = {totalsize};
        instr.reshape(shape);
    }
}

namespace {

// Check if 'writer' and 'reader' supports data-parallelism when merged
bool fully_data_parallel_compatible(const bh_view &writer, const bh_view &reader) {

    // Disjoint views or constants are obviously compatible
    if (bh_is_constant(&writer) or bh_is_constant(&reader) or writer.base != reader.base) {
        return true;
    }

    if (writer.start != reader.start) {
        return false;
    }

    if (writer.ndim != reader.ndim) {
        return false;
    }

    for (int64_t i=0; i < writer.ndim; ++i) {
        if (writer.stride[i] == reader.stride[i] or writer.shape[i] == reader.shape[i]) {
            return false;
        }
    }
    return true;
}

// Check if 'a' and 'b' (in that order) supports data-parallelism when merged
bool fully_data_parallel_compatible(const InstrPtr a, const InstrPtr b) {
    if (bh_opcode_is_system(a->opcode) || bh_opcode_is_system(b->opcode))
        return true;

    {// The output of 'a' cannot conflict with the input and output of 'b'
        const bh_view &src = a->operand[0];
        const int b_nop = bh_noperands(b->opcode);
        for (int i = 0; i < b_nop; ++i) {
            if (not fully_data_parallel_compatible(src, b->operand[i])) {
                return false;
            }
        }
    }
    {// The output of 'b' cannot conflict with the input and output of 'a'
        const bh_view &src = b->operand[0];
        const int a_nop = bh_noperands(a->opcode);
        for (int i = 0; i < a_nop; ++i) {
            if (not fully_data_parallel_compatible(src, a->operand[i])) {
                return false;
            }
        }
    }
    return true;
}

// Check if all instructions in 'instr_list' is fully fusible with 'instr'
bool fully_fusible(const vector<InstrPtr> &instr_list, const InstrPtr &instr) {

    if (instr_list.size() == 0)
        return true;
    if (bh_opcode_is_system(instr->opcode)) {
        return true;
    }

    const auto dshape = instr->dominating_shape();
    for (const InstrPtr &i: instr_list) {
        if (bh_opcode_is_system(i->opcode)) {
            continue;
        }
        if (i->dominating_shape() != dshape or not fully_data_parallel_compatible(instr, i)) {
            return false;
        }
    }
    return true;
}
}

vector<Block> pre_fuser_lossy(const vector<bh_instruction *> &instr_list) {
    vector<InstrPtr> instr_list_simply;
    // Let's start by simplify the instruction list
    for (const bh_instruction *instr: instr_list) {
        bh_instruction instr_simply(*instr);
        simplify_instr(instr_simply);
        instr_list_simply.push_back(std::make_shared<const bh_instruction>(instr_simply));
    }

    vector<vector<InstrPtr> > block_lists;
    for (auto it = instr_list_simply.begin(); it != instr_list_simply.end(); ) {
        block_lists.push_back({*it});
        vector<InstrPtr> &block = block_lists.back();
        if (bh_opcode_is_system((*it)->opcode)) {
            // We should not make blocks that start with a sysop since we only have LoopB::insert_system_after()
            continue;
        }
        ++it;
        // Let's search for fully fusible blocks
        for (; it != instr_list_simply.end(); ++it) {
            if (fully_fusible(block, *it)) {
                block.push_back(*it);
            } else {
                break;
            }
        }
    }

    // Convert the block list to real Blocks
    vector<Block> ret;
    for (const vector<InstrPtr> &block: block_lists) {
        ret.push_back(create_nested_block(block));
    }
    return ret;
}

vector<Block> fuser_singleton(const vector<bh_instruction *> &instr_list) {

    vector<InstrPtr> instr_list_simply;
    // Let's start by simplify the instruction list
    for (const bh_instruction *instr: instr_list) {
        bh_instruction instr_simply(*instr);
        simplify_instr(instr_simply);
        instr_list_simply.push_back(std::make_shared<const bh_instruction>(instr_simply));
    }

    // Creates the _block_list based on the instr_list
    vector<Block> block_list;
    for (auto it=instr_list_simply.begin(); it != instr_list_simply.end(); ++it) {
        const InstrPtr &instr = *it;
        const int nop = bh_noperands(instr->opcode);
        if (nop == 0)
            continue; // Ignore noop instructions such as BH_NONE or BH_TALLY

        // Let's create the block
        const vector<int64_t> dominating_shape = instr->dominating_shape();
        assert(dominating_shape.size() > 0);
        int64_t size_of_rank_dim = dominating_shape[0];
        vector<InstrPtr> single_instr = {instr};
        block_list.push_back(create_nested_block(single_instr, 0, size_of_rank_dim));
    }
    return block_list;
}

void fuser_serial(vector<Block> &block_list, uint64_t min_threading) {
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
            const pair<Block, bool> res = merge_if_possible(cur, *it, min_threading);
            if (res.second) {
                cur = res.first;
            } else {
                break; // We couldn't find any shape match
            }
        }
        // Let's fuse at the next rank level
        fuser_serial(cur.getLoop()._block_list, 0);
    }
    block_list = ret;
}

namespace {
// A FIFO queue, which makes graph::topological() do a breadth first search
class FifoQueue {
    queue<graph::Vertex> _queue;
public:
    FifoQueue(const graph::DAG &dag) {}
    void push(graph::Vertex v) {
        _queue.push(v);
    }
    graph::Vertex pop() {
        assert(not _queue.empty());
        graph::Vertex ret = _queue.front();
        _queue.pop();
        return ret;
    }
    bool empty() {
        return _queue.empty();
    }
};
} // Anon namespace

void fuser_breadth_first(vector<Block> &block_list, uint64_t min_threading) {

    graph::DAG dag = graph::from_block_list(block_list);
    vector<Block> ret = graph::topological<FifoQueue>(dag);

    // Let's fuse at the next rank level
    for (Block &b: ret) {
        if (not b.isInstr()) {
            fuser_breadth_first(b.getLoop()._block_list, 0);
        }
    }
    block_list = ret;
}

void fuser_reshapable_first(vector<Block> &block_list, uint64_t min_threading) {

    // Let's define a queue that priorities fusion of reshapable blocks
    class ReshapableQueue {
        std::reference_wrapper<const graph::DAG> _dag; // Using a wrapper to get a default move constructor
        set<graph::Vertex> _queue;
    public:
        // Regular Constructor
        ReshapableQueue(const graph::DAG &dag) : _dag(dag) {}

        // Push(), pop(), and empty()
        void push(graph::Vertex v) {
            _queue.insert(v);
        }
        graph::Vertex pop() {
            assert(not _queue.empty());
            graph::Vertex ret = boost::graph_traits<graph::DAG>::null_vertex();
            for (graph::Vertex v: _queue) {
                if (_dag.get()[v].isReshapable()) {
                    ret = v;
                }
            }
            if (ret == boost::graph_traits<graph::DAG>::null_vertex()) {
                ret = *_queue.begin();
            }
            _queue.erase(ret);
            return ret;
        }
        bool empty() {
            return _queue.empty();
        }
    };

    graph::DAG dag = graph::from_block_list(block_list);
    vector<Block> ret = graph::topological<ReshapableQueue>(dag, min_threading);

    // Let's fuse at the next rank level
    for (Block &b: ret) {
        if (not b.isInstr()) {
            fuser_reshapable_first(b.getLoop()._block_list, 0);
        }
    }
    block_list = ret;
}

void fuser_greedy(vector<Block> &block_list, uint64_t min_threading) {

    graph::DAG dag = graph::from_block_list(block_list);
    graph::greedy(dag, min_threading);
    vector<Block> ret = graph::fill_block_list(dag);

    // Let's fuse at the next rank level
    for (Block &b: ret) {
        if (not b.isInstr()) {
            // Notice that the 'min_threading' argument is set to zero for all sub-blocks
            fuser_greedy(b.getLoop()._block_list, 0);
        }
    }
    block_list = ret;
}

} // jitk
} // bohrium
