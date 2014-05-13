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
#include <bh.h>
#include "boost/graph/adjacency_list.hpp"
#include "boost/graph/transitive_reduction.hpp"
#include "boost/graph/topological_sort.hpp"
#include <boost/graph/graphviz.hpp>

namespace bohrium {
namespace core {
namespace dag {

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
    Dag(bh_ir* bhir);

    ~Dag(void);

    /**
     *  Returns a reference to the subgraphs within the graph.
     */
    std::vector<Graph*>& subgraphs(void);

    /**
     *  Returns a textual representation of graph meta-data
     *  such as the number of nodes, edges etc.
     */
    std::string text(void);

    /**
     *  Returns a textual representation in dot-format
     */
    std::string dot(void);

    std::string dot(bh_instruction* instr, int64_t nr);

private:
    /**
     *  Construct dependencies in the adjacency_list.
     */
    void array_deps(void);
    void system_deps(void);

    /**
     *  Partition the graph into subgraphs with certain properties...
     */
    void partition(void);

    bh_ir* _bhir;
    Graph _dag;
    std::vector<Graph*> _subgraphs;

    static const char TAG[];
};

}}}

