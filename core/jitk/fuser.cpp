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

#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/breadth_first_search.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/foreach.hpp>
#include <fstream>
#include <numeric>
#include <queue>

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

// Merges the two blocks 'a' and 'a' (in that order)
pair<Block, bool> block_merge(const Block &a, const Block &b, const set<bh_instruction*> &news) {
    assert(a.validation());
    assert(b.validation());

    // First we check for data incompatibility
    if (a.isInstr() or b.isInstr() or not data_parallel_compatible(a, b) or sweeps_accessed_by_block(a._sweeps, b)) {
        return make_pair(Block(), false);
    }
    // Check for perfect match, which is directly mergeable
    if (a.size == b.size) {
        return make_pair(merge(a, b), true);
    }

    // System-only blocks are very flexible because they array sizes does not have to match when reshaping
    // thus we can simply prepend/append system instructions without further checks.
    if (b.isSystemOnly()) {
        Block block(a);
        assert(block.validation());
        block.append_instr_list(b.getAllInstr());
        assert(block.validation());
        return make_pair(block, true);
    } else if (a.isSystemOnly()){
        Block block(b);
        assert(block.validation());
        block.prepend_instr_list(a.getAllInstr());
        assert(block.validation());
        return make_pair(block, true);
    }

    // Check fusibility of reshapable blocks
    if (b._reshapable && b.size % a.size == 0) {
        vector<bh_instruction *> cur_instr = a.getAllInstr();
        vector<bh_instruction *> it_instr = b.getAllInstr();
        cur_instr.insert(cur_instr.end(), it_instr.begin(), it_instr.end());
        return make_pair(create_nested_block(cur_instr, b.rank, a.size, news), true);
    }
    if (a._reshapable && a.size % b.size == 0) {
        vector<bh_instruction *> cur_instr = a.getAllInstr();
        vector<bh_instruction *> it_instr = b.getAllInstr();
        cur_instr.insert(cur_instr.end(), it_instr.begin(), it_instr.end());
        return make_pair(create_nested_block(cur_instr, a.rank, b.size, news), true);
    }
    return make_pair(Block(), false);
}

vector<Block> fuser_serial(const vector<Block> &block_list, const set<bh_instruction*> &news) {
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
            const pair<Block, bool> res = block_merge(cur, *it, news);
            if (res.second) {
                cur = res.first;
            } else {
                break; // We couldn't find any shape match
            }
        }
        // Let's fuse at the next rank level
        cur._block_list = fuser_serial(cur._block_list, news);
    }
    return ret;
}

namespace dag {

//The type declaration of the boost graphs, vertices and edges.
typedef boost::adjacency_list<boost::setS, boost::vecS, boost::bidirectionalS, const Block*> DAG;
typedef typename boost::graph_traits<DAG>::edge_descriptor Edge;
typedef uint64_t Vertex;

/* Determines whether there exist a path from 'a' to 'b'
 *
 * Complexity: O(E + V)
 *
 * @a               The first vertex
 * @b               The second vertex
 * @dag             The DAG
 * @only_long_path  Only accept path of length greater than one
 * @return          True if there is a path
 */
bool path_exist(Vertex a, Vertex b, const DAG &dag, bool only_long_path) {
    using namespace boost;

    struct path_visitor:default_bfs_visitor {
        const Vertex dst;
        path_visitor(Vertex b):dst(b){};

        void examine_edge(Edge e, const DAG &g) const {
            if(target(e,g) == dst)
                throw runtime_error("");
        }
    };
    struct long_visitor:default_bfs_visitor {
        const Vertex src, dst;
        long_visitor(Vertex a, Vertex b):src(a),dst(b){};

        void examine_edge(Edge e, const DAG &g) const
        {
            if(source(e,g) != src and target(e,g) == dst)
                throw runtime_error("");
        }
    };
    try {
        if(only_long_path)
            breadth_first_search(dag, a, visitor(long_visitor(a,b)));
        else
            breadth_first_search(dag, a, visitor(path_visitor(b)));
    }
    catch (const runtime_error &e) {
        return true;
    }
    return false;
}

// Create a DAG based on the 'block_list'
DAG from_block_list(const vector<Block> &block_list) {
    DAG graph;
    map<bh_base*, set<Vertex> > base2vertices;
    for (const Block &block: block_list) {
        assert(block.validation());
        Vertex vertex = boost::add_vertex(&block, graph);

        // Find all vertices that must connect to 'vertex'
        // using and updating 'base2vertices'
        set<Vertex> connecting_vertices;
        for (bh_base *base: block.getAllBases()) {
            set<Vertex> &vs = base2vertices[base];
            connecting_vertices.insert(vs.begin(), vs.end());
            vs.insert(vertex);
        }

        // Finally, let's add edges to 'vertex'
        BOOST_REVERSE_FOREACH (Vertex v, connecting_vertices) {
            if (vertex != v and block.depend_on(*graph[v])) {
                boost::add_edge(v, vertex, graph);
            }
        }
    }
    return graph;
}

// Pretty print the DAG. A "-<id>.dot" is append the filename.
void pprint(const DAG &dag, const string &filename) {

    //We define a graph and a kernel writer for graphviz
    struct graph_writer {
        const DAG &graph;
        graph_writer(const DAG &g) : graph(g) {};
        void operator()(std::ostream& out) const {
            out << "graph [bgcolor=white, fontname=\"Courier New\"]" << endl;
            out << "node [shape=box color=black, fontname=\"Courier New\"]" << endl;
        }
    };
    struct kernel_writer {
        const DAG &graph;
        kernel_writer(const DAG &g) : graph(g) {};
        void operator()(std::ostream& out, const Vertex& v) const {
            out << "[label=\"Kernel " << v;

            out << ", Instructions: \\l";
            for (const bh_instruction *instr: graph[v]->getAllInstr()) {
                out << *instr << "\\l";
            }
            out << "\"]";
        }
    };
    struct edge_writer {
        const DAG &graph;
        edge_writer(const DAG &g) : graph(g) {};
        void operator()(std::ostream& out, const Edge& e) const {

        }
    };

    static int count=0;
    stringstream ss;
    ss << filename << "-" << count++ << ".dot";
    ofstream file;
    cout << ss.str() << endl;
    file.open(ss.str());
    boost::write_graphviz(file, dag, kernel_writer(dag), edge_writer(dag), graph_writer(dag));
    file.close();
}

vector<Block> breadth_first(DAG &dag, const set<bh_instruction*> &news) {
    vector<Block> ret;
    queue<Vertex> roots; // The root vertices
    // Initiate 'roots'
    BOOST_FOREACH (Vertex v, boost::vertices(dag)) {
        if (boost::in_degree(v, dag) == 0) {
            roots.push(v);
        }
    }

    while (not roots.empty()) { // Each iteration creates a new block
        const Vertex vertex = roots.front(); roots.pop();
        ret.emplace_back(*dag[vertex]);
        Block &block = ret.back();

        // Add adjacent vertices and remove the block from 'dag'
        BOOST_FOREACH (const Vertex v, boost::adjacent_vertices(vertex, dag)) {
            if (boost::in_degree(v, dag) <= 1) {
                roots.push(v);
            }
        }
        boost::clear_vertex(vertex, dag);

        // Instruction blocks should never be merged
        if (block.isInstr()) {
            continue;
        }

        // Roots not fusible with 'block'
        queue<Vertex> nonfusible_roots;
        // Search for fusible blocks within the root blocks
        while (not roots.empty()) {
            const Vertex v = roots.front(); roots.pop();
            const Block &b = *dag[v];
            const pair<Block, bool> res = block_merge(block, b, news);
            if (res.second) {
                block = res.first;
                assert(block.validation());

                // Add adjacent vertices and remove the block 'b' from 'dag'
                BOOST_FOREACH (const Vertex adj, boost::adjacent_vertices(v, dag)) {
                    if (boost::in_degree(adj, dag) <= 1) {
                        roots.push(adj);
                    }
                }
                boost::clear_vertex(v, dag);
            } else {
                nonfusible_roots.push(v);
            }
        }
        roots = nonfusible_roots;
    }
    return ret;
}

} // dag

vector<Block> fuser_breadth_first(const vector<Block> &block_list, const set<bh_instruction *> &news) {

    dag::DAG dag = dag::from_block_list(block_list);
    vector<Block> ret = dag::breadth_first(dag, news);

    // Let's fuse at the next rank level
    for (Block &b: ret) {
        if (not b.isInstr()) {
            b._block_list = fuser_breadth_first(b._block_list, news);
        }
    }
    return ret;
}

} // jitk
} // bohrium
