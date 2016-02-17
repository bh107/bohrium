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

#ifndef __BH_IR_DAG_H
#define __BH_IR_DAG_H

#include <boost/foreach.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <vector>
#include <set>
#include <map>
#include <bh.h>

namespace bohrium {
namespace dag {

typedef uint64_t Vertex;

//The weight class bundled with the weight graph
struct EdgeWeight
{
    int64_t value;
    EdgeWeight(){}
    EdgeWeight(int64_t weight):value(weight){}
};

//The type declaration of the boost graphs, vertices and edges.
typedef boost::adjacency_list<boost::setS, boost::vecS, boost::bidirectionalS,
                              bh_ir_kernel> GraphD;
typedef boost::adjacency_list<boost::setS, boost::vecS, boost::undirectedS,
                              boost::no_property, EdgeWeight> GraphW;
typedef typename boost::graph_traits<GraphD>::edge_descriptor EdgeD;
typedef typename boost::graph_traits<GraphW>::edge_descriptor EdgeW;

//Forward declarations
class GraphDW;
bool path_exist(Vertex a, Vertex b, const GraphD &dag,
                bool ignore_neighbors);
void pprint(const GraphDW &dag, const char filename[]);
bool cycles(const GraphD &g);


/* The GraphDW class encapsulate both a dependency graph and
 * a weight graph. The public methods ensures that the two
 * graphs are synchronized and are always in a valid state.
 */
class GraphDW
{
protected:
    GraphD _bglD;
    GraphW _bglW;

public:
    //Map from base-array to the set of vertices that accesses it
    const GraphD &bglD() const {return _bglD;}
    const GraphW &bglW() const {return _bglW;}

    /* Adds a vertex to both the dependency and weight graph.
     * Additionally, both dependency and weight edges are
     * added / updated as needed.
     *
     * @base2vertices  In order to improve the build process, this
     *                 function accepts and maintain a map from base-
     *                 array to the set of vertices that accesses it
     */
    Vertex add_vertex(const bh_ir_kernel &kernel,
                      std::map<bh_base*,std::set<Vertex> > &base2vertices);

    /* The default constructor */
    GraphDW(){};

    /* Constructor based on a dependency graph.
     *
     * @dag     The dependency graph
     */
    GraphDW(const GraphD &dag);

    /* Add the set of vertices 'sub_graph' to 'this' graph.
     *
     * @sub_graph  The set of vertices in 'dag' to add
     * @dag        The source graph
     */
    void add_from_subgraph(const std::set<Vertex> &sub_graph, const GraphDW &dag);

    /* Removes both weight and dependency edge that connect v1 and v2
     *
     * @v1  Vertex
     * @v2  Vertex
     */
    void remove_edges(Vertex v1, Vertex v2)
    {
        {
            auto e = edge(v1, v2, _bglD);
            if(e.second)
                boost::remove_edge(e.first, _bglD);
        }
        {
            auto e = edge(v2, v1, _bglD);
            if(e.second)
                boost::remove_edge(e.first, _bglD);
        }
        {
            auto e = edge(v1, v2, _bglW);
            if(e.second)
                boost::remove_edge(e.first, _bglW);
        }
    }

    /* Clear the vertex without actually removing it.
     *
     * @v  The Vertex
     */
    void clear_vertex(Vertex v)
    {
        boost::clear_vertex(v, _bglD);
        boost::clear_vertex(v, _bglW);
        _bglD[v].clear();
    }

    /* Remove the previously cleared vertices.
     * NB: invalidates all existing vertex and edge pointers
     *     and iterators
     */
    void remove_cleared_vertices()
    {
        std::vector<Vertex> removes;
        BOOST_FOREACH(Vertex v, boost::vertices(_bglD))
        {
            if(_bglD[v].instr_indexes().size() == 0)
            {
                removes.push_back(v);
            }
        }
        //NB: because of Vertex invalidation, we have to traverse in reverse
        BOOST_REVERSE_FOREACH(Vertex &v, removes)
        {
            boost::remove_vertex(v, _bglD);
            boost::remove_vertex(v, _bglW);
        }
    }

    /* Merge vertex 'a' and 'b' where 'b' is cleared and 'a' becomes
     * the merged vertex.
     *
     * @a          The surviving vertex
     * @b          The cleared vertex
     * @a_before_b Whether to append or prepend the instructions of 'b' to 'a'
     */
    void merge_vertices(Vertex a, Vertex b, bool a_before_b=true);

    /* Merge vertex 'id_a' and 'id_b' where 'id_b' is cleared and 'id_a' becomes
     * the merged vertex.
     *
     * @a          The surviving vertex (as kernel ID)
     * @b          The cleared vertex (as kernel ID)
     */
    void merge_vertices_by_id(uint64_t id_a, uint64_t id_b);

    /* Transitive reduce the 'dag', i.e. remove all redundant edges,
     *
     * Complexity: O(E * (E + V))
     *
     * @a   The first vertex
     * @b   The second vertex
     * @dag The DAG
     */
    void transitive_reduction();
};

/* Creates a new DAG based on a bhir that consist of gently fused
 * instructions.
 * NB: the 'bhir' must not be deallocated or moved before 'dag'
 *
 * Complexity: O(n^2) where 'n' is the number of instructions
 *
 * @bhir  The BhIR
 * @dag   The output dag
 *
 * Throw logic_error() if the kernel_list wihtin 'bhir' isn't empty
 */
void from_bhir(bh_ir &bhir, GraphDW &dag);

/* Creates a new DAG based on a kernel list where each vertex is a kernel.
 * NB: the 'kernels' must not be deallocated or moved before 'dag'.
 *
 * Complexity: O(E + V)
 *
 * @kernels The kernel list
 * @dag     The output dag
 */
void from_kernels(const std::vector<bh_ir_kernel> &kernels, GraphDW &dag);

/* Fills the kernel list based on the DAG where each vertex is a kernel.
 *
 * Complexity: O(E + V)
 *
 * @dag     The dag
 * @kernels The kernel list output
 */
void fill_kernel_list(const GraphD &dag, std::vector<bh_ir_kernel> &kernel_list);

/* Split the 'dag' into sub-graphs that may be handle individually
 *
 * Complexity: O(E + V)
 *
 * @dag     The dag
 * @kernels The vector of output sub-graphs in topological order
 */
void split(const GraphDW &dag, std::vector<GraphDW> &output);

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
bool path_exist(Vertex a, Vertex b, const GraphD &dag, bool only_long_path=false);

/* Determines whether there are cycles in the Graph
 *
 * Complexity: O(E + V)
 *
 * @g       The digraph
 * @return  True if there are cycles in the digraph, else false
 */
bool cycles(const GraphD &g);

/* Determines the cost of the DAG.
 *
 * Complexity: O(E + V)
 *
 * @dag     The DAG
 * @return  The cost
 */
uint64_t dag_cost(const GraphD &dag);

/* Determines the cost of the DAG using the
 * Unique-Views cost model.
 *
 * Complexity: O(E + V)
 *
 * @dag     The DAG
 * @return  The cost
 */
uint64_t dag_cost_unique_views(const GraphD &dag);

/* Sort the weights in descending order
 *
 * Complexity: O(E * log E)
 *
 * @edges  The input/output edge list
 */
void sort_weights(const GraphW &dag, std::vector<EdgeW> &edges);

/* Writes the DOT file of a DAG
 *
 * Complexity: O(E + V)
 *
 * @dag       The DAG to write
 * @filename  The name of DOT file
 */
void pprint(const GraphDW &dag, const char filename[]);

/* Check that the 'dag' is valid
 *
 * @dag                   The dag in question
 * @transitivity_allowed  Is transitive edges allowed in the dag?
 * @return                The bool answer
 */
bool dag_validate(const GraphDW &dag, bool transitivity_allowed=true);

/* Check that the vector of 'dags' is valid
 *
 * @bhir                  The BhIR in question
 * @dags                  The vector of dags in question
 * @transitivity_allowed  Is transitive edges allowed in the dag?
 * @return                The bool answer
 */
bool dag_validate(const bh_ir &bhir, const std::vector<GraphDW> &dags, bool transitivity_allowed=true);


/* Returns the set of non-fusibles for each vertex in 'dag'
 *
 * @dag                   The dag in question
 * @return                The vertex-to-non-fusibles map
 */
std::map<Vertex, std::set<Vertex> > get_vertex2nonfusibles(const GraphD &dag);

/* Fuse vertices in the graph that can be fused without
 * changing any future possible fusings
 * NB: invalidates all existing vertex and edge pointers.
 *
 * Complexity: O(E * (E + V))
 *
 * @dag The DAG to fuse
 */
void fuse_gently(GraphDW &dag);

/* Fuse vertices in the graph greedily, which is a non-optimal
 * algorithm that fuses the most costly edges in the DAG first.
 * The edges in 'ignores' will not be merged.
 * NB: invalidates all existing edge iterators.
 *
 * Complexity: O(E^2 * (E + V))
 *
 * @dag      The DAG to fuse
 * @ignores  List of edges not to merge
 */
void fuse_greedy(GraphDW &dag);
void fuse_greedy(GraphDW &dag, const std::set<Vertex> *ignores);

}} //namespace bohrium::dag

#endif

