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
#ifndef __BH_CORE_DAG
#define __BH_CORE_DAG

#include <bh.h>
#include <boost/graph/subgraph.hpp>
#include <boost/graph/adjacency_list.hpp>
#include "tac.h"
#include "symbol_table.hpp"

namespace bohrium {
namespace core {

// Underlying graph representation and implementation
typedef boost::adjacency_list_traits<boost::vecS, boost::vecS, boost::directedS> Traits;

// Graph representation
typedef boost::subgraph< boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS,
    boost::property<boost::vertex_color_t, int>, boost::property<boost::edge_index_t, int> > > Graph;

// Iterating over vertices and edges
typedef boost::graph_traits<Graph>::vertex_iterator vertex_iter;
typedef boost::graph_traits<Graph>::edge_iterator edge_iter;

class Dag
{
public:
    /**
     *  Construct a graph with instructions as vertices and edges as data-dependencies.
     *  @param bhir The bhir containing list of instructions.
     */
    Dag(bh_instruction* instrs, bh_intp ninstrs);

    /**
     * Deconstructor... not much to say here...
     */
    ~Dag(void);

    /**
     *  Return a reference to the graph.
     */
    Graph& graph(void);

    /**
     *  Returns a reference to the subgraphs within the graph.
     */
    std::vector<Graph*>& subgraphs(void);

    /**
     *  Return the requested bh_instruction.
     */
    bh_instruction& instr(size_t instr_idx);

    /**
     *  Returns the requested three-address-code.
     */
    tac_t& tac(size_t tac_idx);

    /**
     *  Returns a textual representation of graph meta-data
     *  such as the number of nodes, edges etc.
     */
    std::string text(void);

    /**
     *  Returns a textual representation in dot-format of the program.
     */
    std::string dot(void);

    /**
     *  Return a textual representation in dot-format of a tac_t.
     */
    std::string dot(tac_t& tac, int64_t nr);

    /**
     *  Return a textual representation in dot-format of a bh_instruction.
     */
    std::string dot(bh_instruction* instr, int64_t nr);

private:
    /**
     *  Construct dependencies in the adjacency_list based on array operations.
     */
    void array_deps(void);

    /**
     * Construct dependencies in the graph based on system operations.
     */
    void system_deps(void);

    /**
     *  Partition the graph into subgraphs with certain properties...
     */
    void partition(void);

    /**
     *  Determine whether two instructions are fusable.
     */
    bool fusable(tac_t& prev, tac_t& cur);

    bh_instruction* instr_;         // Array of instructions
    bh_intp ninstr_;                // The number of instructions
    SymbolTable symbol_table_;      // Symbols of operands
    std::vector<tac_t> tacs_;       // Three-address-code format of instructions
    Graph graph_;                   // Graph form of instructions
    std::vector<Graph*> subgraphs_; // Partitioning of the graph in subgraphs

    static const char TAG[];
};

}}

#endif
