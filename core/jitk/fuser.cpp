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


vector<Block> fuser_singleton(const vector<bh_instruction *> &instr_list) {

    // Creates the _block_list based on the instr_list
    vector<Block> block_list;
    for (auto it=instr_list.begin(); it != instr_list.end(); ++it) {
        bh_instruction instr(**it);
        int nop = bh_noperands(instr.opcode);
        if (nop == 0)
            continue; // Ignore noop instructions such as BH_NONE or BH_TALLY

        // Let's try to simplify the shape of the instruction
        if (instr.reshapable()) {
            const vector<int64_t> dominating_shape = instr.dominating_shape();
            assert(dominating_shape.size() > 0);

            const int64_t totalsize = std::accumulate(dominating_shape.begin(), dominating_shape.end(), 1, \
                                                      std::multiplies<int64_t>());
            const vector<int64_t> shape = {totalsize};
            instr.reshape(shape);
        }
        // Let's create the block
        const vector<int64_t> dominating_shape = instr.dominating_shape();
        assert(dominating_shape.size() > 0);
        int64_t size_of_rank_dim = dominating_shape[0];
        vector<InstrPtr> single_instr = {std::make_shared<bh_instruction>(instr)};
        block_list.push_back(create_nested_block(single_instr, 0, size_of_rank_dim));
    }
    return block_list;
}

void fuser_serial(vector<Block> &block_list) {
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
            const pair<Block, bool> res = merge_if_possible(cur, *it);
            if (res.second) {
                cur = res.first;
            } else {
                break; // We couldn't find any shape match
            }
        }
        // Let's fuse at the next rank level
        fuser_serial(cur.getLoop()._block_list);
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

void fuser_breadth_first(vector<Block> &block_list) {

    graph::DAG dag = graph::from_block_list(block_list);
    vector<Block> ret = graph::topological<FifoQueue>(dag);

    // Let's fuse at the next rank level
    for (Block &b: ret) {
        if (not b.isInstr()) {
            fuser_breadth_first(b.getLoop()._block_list);
        }
    }
    block_list = ret;
}

void fuser_reshapable_first(vector<Block> &block_list) {

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
    vector<Block> ret = graph::topological<ReshapableQueue>(dag);

    // Let's fuse at the next rank level
    for (Block &b: ret) {
        if (not b.isInstr()) {
            fuser_reshapable_first(b.getLoop()._block_list);
        }
    }
    block_list = ret;
}

void fuser_greedy(vector<Block> &block_list) {

    graph::DAG dag = graph::from_block_list(block_list);
    graph::greedy(dag);
    vector<Block> ret = graph::topological<FifoQueue>(dag);

    // Let's fuse at the next rank level
    for (Block &b: ret) {
        if (not b.isInstr()) {
            fuser_greedy(b.getLoop()._block_list);
        }
    }
    block_list = ret;
}

} // jitk
} // bohrium
