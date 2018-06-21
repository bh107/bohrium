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
#include <boost/graph/topological_sort.hpp>
#include <boost/foreach.hpp>
#include <fstream>
#include <numeric>
#include <queue>
#include <cassert>

#include <jitk/graph.hpp>
#include <jitk/block.hpp>

using namespace std;

namespace bohrium {
namespace jitk {
namespace graph {

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
    map<const bh_base*, set<Vertex> > base2vertices;
    for (const Block &block: block_list) {
        assert(block.validation());
        Vertex vertex = boost::add_vertex(block, graph);

        // Find all vertices that must connect to 'vertex'
        // using and updating 'base2vertices'
        set<Vertex> connecting_vertices;
        for (const bh_base *base: block.getAllBases()) {
            set<Vertex> &vs = base2vertices[base];
            connecting_vertices.insert(vs.begin(), vs.end());
            vs.insert(vertex);
        }
        // Let's add edges to 'vertex'
        BOOST_REVERSE_FOREACH (Vertex v, connecting_vertices) {
            if (vertex != v and block.dependOn(graph[v])) {
                boost::add_edge(v, vertex, graph);
            }
        }
        // Then we do the same for dependency because of freed arrays
        set<Vertex> connecting_vertices_freed;
        if (not block.isInstr()) {
            for (const bh_base *base: block.getLoop().getAllFrees()) {
                set<Vertex> &vs = base2vertices[base];
                connecting_vertices_freed.insert(vs.begin(), vs.end());
                vs.insert(vertex);
            }
        }
        BOOST_REVERSE_FOREACH (Vertex v, connecting_vertices_freed) {
            if (vertex != v) {
                boost::add_edge(v, vertex, graph);
            }
        }
    }
    return graph;
}

vector<Block> fill_block_list(const DAG &dag) {
    vector<Block> ret;
    vector<Vertex> topological_order;
    boost::topological_sort(dag, back_inserter(topological_order));
    BOOST_REVERSE_FOREACH(const Vertex &v, topological_order) {
        ret.push_back(dag[v]);
    }
    return ret;
}

uint64_t weight(const Block &b1, const Block &b2) {
    if (b1.isInstr() or b2.isInstr()) {
        return 0; // Instruction blocks cannot be fused
    }
    const set<bh_base *> news = b1.getLoop().getAllNews();
    const set<bh_base *> frees = b2.getLoop().getAllFrees();
    vector<bh_base *> new_temps;
    set_intersection(news.begin(), news.end(), frees.begin(), frees.end(), back_inserter(new_temps));

    uint64_t totalsize = 0;
    for (const bh_base *base: new_temps) {
        totalsize += base->nbytes();
    }
    return totalsize;
}

uint64_t block_cost(const Block &block) {
    std::vector<bh_base*> non_temps;
    const set<bh_base *> temps = block.isInstr()?set<bh_base *>():block.getLoop().getAllTemps();
    for (const InstrPtr instr: block.getAllInstr()) {
        // Find non-temporary arrays
        for(const bh_view &v: instr->operand) {
            if (not bh_is_constant(&v) and temps.find(v.base) == temps.end()) {
                if (std::find(non_temps.begin(), non_temps.end(), v.base) == non_temps.end()) {
                    non_temps.push_back(v.base);
                }
            }
        }
    }
    uint64_t totalsize = 0;
    for (const bh_base *base: non_temps) {
        totalsize += base->nbytes();
    }
    return totalsize;
}

bool validate(DAG &dag) {
    return true;
}

void merge_vertices(DAG &dag, Vertex a, Vertex b, const bool remove_b) {
    // Let's merge the two blocks and save it in vertex 'a'
    assert(not dag[a].isInstr());
    assert(not dag[b].isInstr());
    dag[a] = reshape_and_merge(dag[a].getLoop(), dag[b].getLoop());
    assert(dag[a].validation());

    // Add new children
    BOOST_FOREACH(Vertex child, boost::adjacent_vertices(b, dag)) {
        assert(child != a);
        boost::add_edge(a, child, dag);
    }
    // Add new parents
    BOOST_FOREACH(Vertex parent, boost::inv_adjacent_vertices(b, dag)) {
        if (parent != a) {
            boost::add_edge(parent, a, dag);
        }
    }
    // Finally, cleanup of 'b'
    boost::clear_vertex(b, dag);
    if (remove_b) {
        boost::remove_vertex(b, dag);
    }
    assert(validate(dag));
}

void transitive_reduction(DAG &dag) {
    vector<Edge> removals;
    BOOST_FOREACH(Edge e, boost::edges(dag)) {
        if(path_exist(source(e,dag), target(e,dag), dag, true))
            removals.push_back(e);
    }
    for (Edge &e: removals) {
        remove_edge(e, dag);
    }
    assert(validate(dag));
}

void merge_system_pendants(DAG &dag) {
    // Find edges to merge over
    vector<Edge> merges;
    BOOST_FOREACH(Edge e, boost::edges(dag)) {
        Vertex src = boost::source(e, dag);
        Vertex dst = boost::target(e, dag);
        if (boost::in_degree(dst, dag) == 1 and boost::out_degree(dst, dag) == 0) { // Leaf
            if (dag[dst].isSystemOnly()) {
                merges.push_back(e);
            }
        } else if (boost::in_degree(src, dag) == 0 and boost::out_degree(src, dag) == 1) { //Root
            if (dag[src].isSystemOnly()) {
                merges.push_back(e);
            }
        }
    }
    // Do merge
    for (Edge &e: merges) {
        Vertex src = boost::source(e, dag);
        Vertex dst = boost::target(e, dag);
        merge_vertices(dag, src, dst, false);
    }
    // Remove the vertex leftover from the merge
    // NB: because of Vertex invalidation, we have to traverse in reverse
    BOOST_REVERSE_FOREACH(Edge &e, merges) {
        boost::remove_vertex(boost::target(e, dag), dag);
    }
    assert(validate(dag));
}

void pprint(const DAG &dag, const char *filename, bool avoid_rank0_sweep, int id) {

    //We define a graph and a kernel writer for graphviz
    struct graph_writer {
        const DAG &graph;
        graph_writer(const DAG &g) : graph(g) {};
        void operator()(std::ostream& out) const {
            uint64_t totalcost = 0;
            BOOST_FOREACH(Vertex v, boost::vertices(graph)) {
                totalcost += block_cost(graph[v]);
            }
            out << "labelloc=\"t\";" << endl;
            out << "label=\"Total cost: " << (double) totalcost;

            // Find "work below par-threshold"
            uint64_t threading_below_threshold=0, totalwork=0;
            BOOST_FOREACH(Vertex v, boost::vertices(graph)) {
                uint64_t total_threading = 0;
                if (not graph[v].isInstr()) {
                    total_threading = parallel_ranks(graph[v].getLoop()).second;
                }
                for (const InstrPtr instr: graph[v].getAllInstr()) {
                    if (bh_opcode_is_system(instr->opcode))
                        continue;
                    if (total_threading < 1000) {
                        threading_below_threshold += bh_nelements(instr->operand[0]);
                    }
                    totalwork += bh_nelements(instr->operand[0]);
                }
            }
            out << ", Work below par-threshold(1000): " << threading_below_threshold / (double)totalwork * 100 << "%";
            out << "\";";
            out << "graph [bgcolor=white, fontname=\"Courier New\"]" << endl;
            out << "node [shape=box color=black, fontname=\"Courier New\"]" << endl;
        }
    };
    struct kernel_writer {
        const DAG &graph;
        kernel_writer(const DAG &g) : graph(g) {};
        void operator()(std::ostream& out, const Vertex& v) const {
            out << "[label=\"Kernel " << v;
            out << ", Cost: " << (double) block_cost(graph[v]);
            out << "], Instructions: \\l" << graph[v].pprint("\\l");
            out << "\"]";
        }
    };
    struct edge_writer {
        const DAG &graph;
        const bool avoid_sweep;
        edge_writer(const DAG &g, bool avoid_sweep) : graph(g), avoid_sweep(avoid_sweep) {};
        void operator()(std::ostream& out, const Edge& e) const {
            Vertex src = source(e, graph);
            Vertex dst = target(e, graph);
            out << "[label=\" ";
            out << (double) weight(graph[src], graph[dst]) << " bytes\"";
            if (not mergeable(graph[src], graph[dst], avoid_sweep)) {
                out << " color=red";
            }
            out << "]";
        }
    };

    static int dag_count=0;
    if (id == -1) {
        id = dag_count;
    }
    stringstream ss;
    ss << filename << "-" << id++ << ".dot";
    ofstream file;
    cout << ss.str() << endl;
    file.open(ss.str());
    boost::write_graphviz(file, dag, kernel_writer(dag), edge_writer(dag, avoid_rank0_sweep), graph_writer(dag));
    file.close();
}

void greedy(DAG &dag, bool avoid_rank0_sweep) {
    while(1) {
        // First we find all fusible edges
        vector<Edge> fusibles;
        {
            auto edges = boost::edges(dag);
            for (auto it = edges.first; it != edges.second;) {
                Edge e = *it; ++it; // NB: we iterate here because boost::remove_edge() invalidates 'it'
                Vertex v1 = source(e, dag);
                Vertex v2 = target(e, dag);
                // Remove transitive edges
                if(path_exist(v1, v2, dag, true)) {
                    boost::remove_edge(e, dag);
                } else {
                    const Block &b1 = dag[v1];
                    const Block &b2 = dag[v2];
                    if (mergeable(b1, b2, avoid_rank0_sweep)) {
                        fusibles.push_back(e);
                    }
                }
            }
        }
        // Any more vertices to fuse?
        if (fusibles.size() == 0) {
            break;
        }

        // Let's find the greatest weight edge.
        Edge greatest = fusibles.front();
        uint64_t greatest_weight = weight(dag[source(greatest, dag)], dag[target(greatest, dag)]);
        for (Edge e: fusibles) {
            const uint64_t w = weight(dag[source(e, dag)], dag[target(e, dag)]);
            if (w > greatest_weight) {
                greatest = e;
                greatest_weight = w;
            }
        }
        Vertex v1 = source(greatest, dag);
        Vertex v2 = target(greatest, dag);
//        cout << "merge: " << v1 << ", " << v2 << endl;

        assert(not path_exist(v1, v2, dag, true)); // Transitive edges should have been removed by now

        merge_vertices(dag, v1, v2, true);
    }
    assert(validate(dag));
}

} // graph
} // jitk
} // bohrium
