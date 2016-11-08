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

#ifndef __BH_JITK_GRAPH_HPP
#define __BH_JITK_GRAPH_HPP

#include <set>
#include <vector>
#include <string>

#include <jitk/block.hpp>
#include <bh_instruction.hpp>

#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/foreach.hpp>

namespace bohrium {
namespace jitk {
namespace graph {

//The type declaration of the boost graphs, vertices and edges.
typedef boost::adjacency_list<boost::setS, boost::vecS, boost::bidirectionalS, Block> DAG;
typedef typename boost::graph_traits<DAG>::edge_descriptor Edge;
typedef uint64_t Vertex;

// Pretty print the DAG. A "-<id>.dot" is append the filename.
void pprint(const DAG &dag, const std::string &filename);

// Create a dag based on the 'block_list'
DAG from_block_list(const std::vector <Block> &block_list);

// Merges the vertices in 'dag' topologically using 'Queue' as the Vertex queue.
// 'Queue' is a collection of 'Vertex' that is constructed with the DAG and supports push(), pop(), and empty()
template <typename Queue>
std::vector<Block> topological(DAG &dag, const std::set<bh_instruction*> &news) {
    using namespace std;
    vector<Block> ret;
    Queue roots(dag); // The root vertices

    // Initiate 'roots'
    BOOST_FOREACH (Vertex v, boost::vertices(dag)) {
        if (boost::in_degree(v, dag) == 0) {
            roots.push(v);
        }
    }

    // Each iteration creates a new block
    while (not roots.empty()) {
        const Vertex vertex = roots.pop();
        ret.emplace_back(dag[vertex]);
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
        Queue nonfusible_roots(dag);
        // Search for fusible blocks within the root blocks
        while (not roots.empty()) {
            const Vertex v = roots.pop();
            const Block &b = dag[v];
            const pair<Block, bool> res = merge_if_possible(block, b, news);
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
        roots = std::move(nonfusible_roots);
    }
    return ret;
}

} // graph
} // jit
} // bohrium

#endif
