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
#pragma once

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

// Validate the 'dag'
bool validate(DAG &dag);

/* Merge vertices 'a' and 'b' (in that order) into 'a'
 * If 'remove_b==false' than the 'b' vertex is only cleared not removed from graph 'dag'.
 * NB: 'a' and 'b' MUST be fusible
 *
 * 'min_threading' is the minimum amount of threading acceptable in the merged block (ignored if
 * neither 'a' or 'b' have the requested amount)
 *
 * Complexity: O(V)
 *
 */
void merge_vertices(DAG &dag, Vertex a, Vertex b, const bool remove_b=true);

/* Transitive reduce the 'dag', i.e. remove all redundant edges,
 *
 * Complexity: O(E * (E + V))
 *
 */
void transitive_reduction(DAG &dag);

// Merge pendant vertices that are system only
void merge_system_pendants(DAG &dag);

// Pretty print the DAG. A "-<id>.dot" is append the filename. Set `id` to -1 for a default unique id
void pprint(const DAG &dag, const char *filename, bool avoid_rank0_sweep, int id=-1);

// Create a dag based on the 'block_list'
DAG from_block_list(const std::vector <Block> &block_list);

// Create a block list based on the 'dag'
std::vector<Block> fill_block_list(const DAG &dag);

// Merges the vertices in 'dag' topologically using 'Queue' as the Vertex queue.
// 'Queue' is a collection of 'Vertex' that is constructed with the DAG and supports push(), pop(), and empty()
// 'avoid_rank0_sweep' will avoid fusion of sweeped and non-sweeped blocks at the root level
template <typename Queue>
std::vector<Block> topological(DAG &dag, bool avoid_rank0_sweep) {
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
            if (!dag[v].isInstr() and mergeable(block, dag[v], avoid_rank0_sweep)) {
                block = reshape_and_merge(block.getLoop(), dag[v].getLoop());
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

// Merges the vertices in 'dag' greedily.
// 'avoid_rank0_sweep' will avoid fusion of sweeped and non-sweeped blocks at the root level
void greedy(DAG &dag, bool avoid_rank0_sweep);

} // graph
} // jit
} // bohrium
