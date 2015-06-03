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
    const GraphD &bglD() const {return _bglD;}
    const GraphW &bglW() const {return _bglW;}

    /* Adds a vertex to both the dependency and weight graph.
     * Additionally, both dependency and weight edges are
     * added / updated as needed.
     *
     * @kernel  The kernel to bundle with the new vertex
     */
    Vertex add_vertex(const bh_ir_kernel &kernel)
    {
        Vertex d = boost::add_vertex(kernel, _bglD);
        Vertex w = boost::add_vertex(_bglW);
        assert(w == d);

        //Add edges
        BOOST_REVERSE_FOREACH(Vertex v, vertices(_bglD))
        {
            if(d != v and not path_exist(v, d, _bglD, false))
            {
                bool dependency = false;
                int dep = kernel.dependency(_bglD[v]);
                if(dep)
                {
                    assert(dep == 1);
                    dependency = true;
                    boost::add_edge(v, d, _bglD);
                }
                int64_t cost = kernel.merge_cost_savings(_bglD[v]);
                if((cost > 0) or (cost == 0 and dependency))
                {
                    boost::add_edge(v, d, EdgeWeight(cost), _bglW);
                }
            }
        }
        return d;
    }

    /* The default and the copy constructors */
    GraphDW(){};
    GraphDW(const GraphDW &graph):_bglD(graph.bglD()), _bglW(graph.bglW()) {};

    /* Constructor based on a dependency graph. All weights are zero.
     *
     * @dag     The dependency graph
     */
    GraphDW(const GraphD &dag)
    {
        _bglD = dag;
        _bglW = GraphW(boost::num_vertices(dag));
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
            if(_bglD[v].instr_indexes.size() == 0)
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

    /* Merge vertex 'a' and 'b' by appending 'b's instructions to 'a'.
     * Vertex 'b' is cleared rather than removed thus existing vertex
     * and edge pointers are still valid after the merge.
     *
     * @a  The first vertex
     * @b  The second vertex
     */
    void merge_vertices(Vertex a, Vertex b)
    {
        using namespace std;
        using namespace boost;

        //Append the instructions of 'b' to 'a'
        BOOST_FOREACH(uint64_t idx, _bglD[b].instr_indexes)
        {
            _bglD[a].add_instr(idx);
        }

        //Add edges of 'b' to 'a'
        BOOST_FOREACH(const Vertex &v, adjacent_vertices(b, _bglD))
        {
            if(a != v)
            {
                add_edge(a, v, _bglD);
                add_edge(a, v, _bglW);
            }
        }
        BOOST_FOREACH(const Vertex &v, inv_adjacent_vertices(b, _bglD))
        {
            if(a != v)
            {
                add_edge(v, a, _bglD);
                add_edge(a, v, _bglW);
            }
        }
        BOOST_FOREACH(const Vertex &v, adjacent_vertices(b, _bglW))
        {
            if(a != v)
            {
                add_edge(a, v, _bglW);
            }
        }
        clear_vertex(b);

        //Update the edge weights of 'a'
        //Note that the 'out_edge_iterator' is invalidated if it points
        //to 'e' and 'e' is removed thus we cannot use BOOST_FOREACH.
        {
            graph_traits<GraphW>::out_edge_iterator it, end;
            tie(it, end) = out_edges(a, _bglW);
            while(it != end)
            {
                Vertex v1 = source(*it, _bglW);
                Vertex v2 = target(*it, _bglW);
                int64_t cost = _bglD[v1].merge_cost_savings(_bglD[v2]);
                if(cost > 0)
                {
                    _bglW[*it++].value = cost;
                }
                else
                {
                    EdgeW e = *it++;
                    remove_edge(e, _bglW);
                }
            }
        }

        //Lets run some unittests
        #ifndef NDEBUG
            BOOST_FOREACH(const EdgeW &e, edges(_bglW))
            {
                if(not _bglD[source(e, _bglW)].fusible(_bglD[target(e, _bglW)]))
                {
                    cout << "non fusible weight edge!: " << e << endl;
                    assert(1 == 2);
                }
            }
            if(not _bglD[a].fusible())
            {
                cout << "kernel merge " << a << " " << b << endl;
                assert(1 == 2);
            }
            if(cycles(_bglD))
            {
                cout << "kernel merge " << a << " " << b << endl;
                assert(1 == 2);
            }
        #endif
    }

    /* Transitive reduce the 'dag', i.e. remove all redundant edges,
     *
     * Complexity: O(E * (E + V))
     *
     * @a   The first vertex
     * @b   The second vertex
     * @dag The DAG
     */
    void transitive_reduction()
    {
        using namespace std;
        using namespace boost;

        //Remove redundant dependency edges
        {
            vector<EdgeD> removals;
            BOOST_FOREACH(EdgeD e, edges(_bglD))
            {
                if(path_exist(source(e,_bglD), target(e,_bglD), _bglD, true))
                    removals.push_back(e);
            }
            BOOST_FOREACH(EdgeD e, removals)
            {
                remove_edge(e, _bglD);
            }
        }
        //Remove redundant weight edges
        {
            vector<EdgeW> removals;
            BOOST_FOREACH(EdgeW e, edges(_bglW))
            {
                Vertex a = source(e, _bglW);
                Vertex b = target(e, _bglW);
                if(edge(a, b, _bglD).second or edge(b, a, _bglD).second)
                    continue;//'a' and 'b' are adjacent in the DAG

                //Remove the edge if 'a' and 'b' are connected in the DAG
                if(path_exist(a, b, _bglD, true) or path_exist(b, a, _bglD, true))
                    removals.push_back(e);
            }
            BOOST_FOREACH(EdgeW e, removals)
            {
                remove_edge(e, _bglW);
            }
        }
    }
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

/* Determines whether there exist a path from 'a' to 'b'
 *
 * Complexity: O(E + V)
 *
 * @a          The first vertex
 * @b          The second vertex
 * @dag        The DAG
 * @long_path  Whether to accept path of length one
 * @return     True if there is a path
 */
bool path_exist(Vertex a, Vertex b, const GraphD &dag, bool long_path=false);

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
 * @dag     The dag in question
 * @return  The bool answer
 */
bool dag_validate(const GraphD &dag);

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
void fuse_greedy(GraphDW &dag, const std::set<Vertex> *ignores=NULL);

}} //namespace bohrium::dag

#endif

