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
#include <boost/graph/topological_sort.hpp>
#include <boost/graph/breadth_first_search.hpp>
#include <boost/graph/graphviz.hpp>
#include <iterator>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <set>
#include <stdexcept>
#include <bh.h>
#include "bh_fuse.h"

namespace bohrium {
namespace dag {

typedef uint64_t Vertex;

/* Returns the cost of a bh_view */
inline static uint64_t cost_of_view(const bh_view &v)
{
    return bh_nelements_nbcast(&v) * bh_type_size(v.base->type);
}

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
     * NB: invalidates all existing edge iterators
     *     but NOT pointers to neither vertices nor edges.
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
     * NB: invalidates all existing edge iterators.
     *
     * @a  The first vertex
     * @b  The second vertex
     */
    void merge_vertices(Vertex a, Vertex b)
    {
        using namespace std;
        using namespace boost;

        BOOST_FOREACH(uint64_t idx, _bglD[b].instr_indexes)
        {
            _bglD[a].add_instr(idx);
        }
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

        vector<EdgeW> edges2remove;
        BOOST_FOREACH(const EdgeW &e, out_edges(a, _bglW))
        {
            Vertex v1 = source(e, _bglW);
            Vertex v2 = target(e, _bglW);
            if(path_exist(v1, v2, _bglD, false))
            {
                int64_t cost = _bglD[v2].merge_cost_savings(_bglD[v1]);
                if(cost >= 0)
                    _bglW[e].value = cost;
                else
                    edges2remove.push_back(e);
            }
            else if(path_exist(v2, v1, _bglD, false))
            {
                int64_t cost = _bglD[v1].merge_cost_savings(_bglD[v2]);
                if(cost >= 0)
                    _bglW[e].value = cost;
                else
                    edges2remove.push_back(e);
            }
            else
            {
                int64_t cost = _bglD[v1].merge_cost_savings(_bglD[v2]);
                if(cost > 0)
                    _bglW[e].value = cost;
                else
                    edges2remove.push_back(e);
            }
        }
        //In order not to invalidate the 'out_edges' iterator, we have
        //to delay the edge removals to this point.
        BOOST_FOREACH(const EdgeW &e, edges2remove)
        {
            remove_edge(e, _bglW);
        }

        //TODO: for now we run some unittests
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
    }

    /* Transitive reduce the 'dag', i.e. remove all redundant edges,
     * NB: invalidates all existing edge iterators.
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

/* Creates a new DAG based on a bhir that consist of single
 * instruction kernels.
 * NB: the 'bhir' must not be deallocated or moved before 'dag'
 *
 * Complexity: O(n^2) where 'n' is the number of instructions
 *
 * @bhir  The BhIR
 * @dag   The output dag
 *
 * Throw logic_error() if the kernel_list wihtin 'bhir' isn't empty
 */
void from_bhir(bh_ir &bhir, GraphDW &dag)
{
    using namespace std;
    using namespace boost;

    if(bhir.kernel_list.size() != 0)
    {
        throw logic_error("The kernel_list is not empty!");
    }
    //Build a singleton DAG
    for(uint64_t idx=0; idx<bhir.instr_list.size(); ++idx)
    {
        bh_ir_kernel k(bhir);
        k.add_instr(idx);
        dag.add_vertex(k);
    }
}

/* Creates a new DAG based on a kernel list where each vertex is a kernel.
 * NB: the 'kernels' must not be deallocated or moved before 'dag'.
 *
 * Complexity: O(E + V)
 *
 * @kernels The kernel list
 * @dag     The output dag
 */
void from_kernels(const std::vector<bh_ir_kernel> &kernels, GraphDW &dag)
{
    using namespace std;
    using namespace boost;

    BOOST_FOREACH(const bh_ir_kernel &kernel, kernels)
    {
        if(kernel.instr_indexes.size() > 0)
            dag.add_vertex(kernel);
    }
}

/* Fills the BhIR's kernel list based on the DAG where each vertex is a kernel.
 *
 * Complexity: O(E + V)
 *
 * @dag     The dag
 * @kernels The kernel list output
 */
void fill_bhir_kernel_list(const GraphD &dag, bh_ir &bhir)
{
    using namespace std;
    using namespace boost;
    assert(bhir.kernel_list.size() == 0);

    vector<Vertex> topological_order;
    topological_sort(dag, back_inserter(topological_order));
    BOOST_REVERSE_FOREACH(const Vertex &v, topological_order)
    {
        if(dag[v].instr_indexes.size() > 0)
        {
            bhir.kernel_list.push_back(dag[v]);
        }
    }
}

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
bool path_exist(Vertex a, Vertex b, const GraphD &dag,
                bool long_path=false)
{
    using namespace std;
    using namespace boost;

    struct path_visitor:default_bfs_visitor
    {
        const Vertex dst;
        path_visitor(Vertex b):dst(b){};

        void examine_edge(EdgeD e, const GraphD &g) const
        {
            if(target(e,g) == dst)
                throw runtime_error("");
        }
    };
    struct long_visitor:default_bfs_visitor
    {
        const Vertex src, dst;
        long_visitor(Vertex a, Vertex b):src(a),dst(b){};

        void examine_edge(EdgeD e, const GraphD &g) const
        {
            if(source(e,g) != src and target(e,g) == dst)
                throw runtime_error("");
        }
    };
    try
    {
        if(long_path)
            breadth_first_search(dag, a, visitor(long_visitor(a,b)));
        else
            breadth_first_search(dag, a, visitor(path_visitor(b)));
    }
    catch (const runtime_error &e)
    {
        return true;
    }
    return false;
}

/* Determines whether there are cycles in the Graph
 *
 * Complexity: O(E + V)
 *
 * @g       The digraph
 * @return  True if there are cycles in the digraph, else false
 */
bool cycles(const GraphD &g)
{
    using namespace std;
    using namespace boost;
    try
    {
        //TODO: topological sort is an efficient method for finding cycles,
        //but we should avoid allocating a vector
        vector<Vertex> topological_order;
        topological_sort(g, back_inserter(topological_order));
        return false;
    }
    catch (const not_a_dag &e)
    {
        return true;
    }
}

/* Determines the cost of the DAG.
 *
 * Complexity: O(E + V)
 *
 * @dag     The DAG
 * @return  The cost
 */
uint64_t dag_cost(const GraphD &dag)
{
    using namespace std;
    using namespace boost;

    uint64_t cost = 0;
    BOOST_FOREACH(const Vertex &v, vertices(dag))
    {
        cost += dag[v].cost();
    }
    return cost;
}

/* Sort the weights in descending order
 *
 * Complexity: O(E * log E)
 *
 * @edges  The input/output edge list
 */
void sort_weights(const GraphW &dag, std::vector<EdgeW> &edges)
{
    struct wcmp
    {
        const GraphW &graph;
        wcmp(const GraphW &d): graph(d){}
        bool operator() (const EdgeW &e1, const EdgeW &e2)
        {
            return (graph[e1].value > graph[e2].value);
        }
    };
    sort(edges.begin(), edges.end(), wcmp(dag));
}

/* Writes the DOT file of a DAG
 *
 * Complexity: O(E + V)
 *
 * @dag       The DAG to write
 * @filename  The name of DOT file
 */
void pprint(const GraphDW &dag, const char filename[])
{
    using namespace std;
    using namespace boost;
    //Lets create a graph with both vertical and horizontal edges
    GraphD new_dag(dag.bglD());
    map<pair<Vertex, Vertex>, pair<int64_t, bool> > weights;

    BOOST_FOREACH(const EdgeW &e, edges(dag.bglW()))
    {
        Vertex src = source(e, dag.bglW());
        Vertex dst = target(e, dag.bglW());
        bool exist = edge(src,dst,new_dag).second or edge(dst,src,new_dag).second;
        if(not exist)
            add_edge(src, dst, new_dag);

        //Save an edge map of weights and if it is directed
        weights[make_pair(src,dst)] = make_pair(dag.bglW()[e].value, exist);
    }

    //We define a graph and a kernel writer for graphviz
    struct graph_writer
    {
        const GraphD &graph;
        graph_writer(const GraphD &g) : graph(g) {};
        void operator()(std::ostream& out) const
        {
            out << "labelloc=\"t\";" << endl;
            out << "label=\"DAG with a total cost of " << dag_cost(graph);
            out << " bytes\";" << endl;
            out << "graph [bgcolor=white, fontname=\"Courier New\"]" << endl;
            out << "node [shape=box color=black, fontname=\"Courier New\"]" << endl;
        }
    };
    struct kernel_writer
    {
        const GraphD &graph;
        kernel_writer(const GraphD &g) : graph(g) {};
        void operator()(std::ostream& out, const Vertex& v) const
        {
            char buf[1024*10];
            out << "[label=\"Kernel " << v << ", cost: " << graph[v].cost();
            out << " bytes\\n";
            out << "Input views: \\l";
            BOOST_FOREACH(const bh_view &i, graph[v].input_list())
            {
                bh_sprint_view(&i, buf);
                out << buf << "\\l";
            }
            out << "Output views: \\l";
            BOOST_FOREACH(const bh_view &i, graph[v].output_list())
            {
                bh_sprint_view(&i, buf);
                out << buf << "\\l";
            }
            out << "Temp base-arrays: \\l";
            BOOST_FOREACH(const bh_base *i, graph[v].temp_list())
            {
                bh_sprint_base(i, buf);
                out << buf << "\\l";
            }
            out << "Instruction list: \\l";
            BOOST_FOREACH(uint64_t idx, graph[v].instr_indexes)
            {
                const bh_instruction &instr = graph[v].bhir->instr_list[idx];
                out << "[" << idx << "] ";
                bh_sprint_instr(&instr, buf, "\\l");
                out << buf << "\\l";
            }
            out << "\"]";
        }
    };
    struct edge_writer
    {
        const GraphD &graph;
        const map<pair<Vertex, Vertex>, pair<int64_t, bool> > &wmap;
        edge_writer(const GraphD &g, const map<pair<Vertex, Vertex>, pair<int64_t, bool> > &w) : graph(g), wmap(w) {};
        void operator()(std::ostream& out, const EdgeD& e) const
        {
            Vertex src = source(e, graph);
            Vertex dst = target(e, graph);
            int64_t c = -1;
            bool directed = true;
            map<pair<Vertex, Vertex>, pair<int64_t, bool> >::const_iterator it = wmap.find(make_pair(src,dst));
            if(it != wmap.end())
                tie(c,directed) = (*it).second;

            out << "[label=\" ";
            if(c == -1)
                out << "N/A\" color=red";
            else
                out << c << " bytes\"";
            if(not directed)
                out << " dir=none color=green constraint=false";
            out << "]";
        }
    };
    ofstream file;
    file.open(filename);
    write_graphviz(file, new_dag, kernel_writer(new_dag),
                   edge_writer(new_dag, weights), graph_writer(new_dag));
    file.close();
}

/* Check that the 'dag' is valid
 *
 * @dag     The dag in question
 * @return  The bool answer
 */
bool dag_validate(const GraphD &dag)
{
    using namespace std;
    using namespace boost;
    BOOST_FOREACH(Vertex v1, vertices(dag))
    {
        BOOST_FOREACH(Vertex v2, vertices(dag))
        {
            if(v1 != v2)
            {
                const int dep = dag[v1].dependency(dag[v2]);
                if(dep == 1)//'v1' depend on 'v2'
                {
                    if(not path_exist(v2, v1, dag, false))
                    {
                        cout << "not path between " << v1 << " and " << v2 << endl;
                        pprint(dag, "validate-fail.dot");
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

/* Fuse vertices in the graph that can be fused without
 * changing any future possible fusings
 * NB: invalidates all existing vertex and edge pointers.
 *
 * Complexity: O(E * (E + V))
 *
 * @dag The DAG to fuse
 */
void fuse_gentle(GraphDW &dag)
{
    using namespace std;
    using namespace boost;

    const GraphD &d = dag.bglD();
    bool not_finished = true;
    while(not_finished)
    {
        not_finished = false;
        BOOST_FOREACH(const EdgeD &e, edges(d))
        {
            const Vertex &src = source(e, d);
            const Vertex &dst = target(e, d);
            if((in_degree(dst, d) == 1 and out_degree(dst, d) == 0) or
               (in_degree(src, d) == 0 and out_degree(src, d) == 1) or
               (in_degree(dst, d) <= 1 and out_degree(src, d) <= 1))
            {
                if(d[dst].fusible_gently(d[src]))
                {
                    dag.merge_vertices(src, dst);
                    not_finished = true;
                    break;
                }
            }
        }
    }
    dag.remove_cleared_vertices();
    dag.transitive_reduction();
}

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
void fuse_greedy(GraphDW &dag, const std::set<Vertex> &ignores={})
{
    using namespace std;
    using namespace boost;

    //Help function to find and sort the weight edges.
    struct
    {
        void operator()(const GraphDW &g, vector<EdgeW> &edge_list,
                        const set<Vertex> &ignores)
        {
            if(ignores.size() == 0)
            {
                BOOST_FOREACH(const EdgeW &e, edges(g.bglW()))
                {
                    edge_list.push_back(e);
                }
            }
            else
            {
                BOOST_FOREACH(const EdgeW &e, edges(g.bglW()))
                {
                    if(ignores.find(source(e, g.bglW())) == ignores.end() and
                       ignores.find(target(e, g.bglW())) == ignores.end())
                        edge_list.push_back(e);
                }
            }
            sort_weights(g.bglW(), edge_list);
        }
    }get_sorted_edges;

    vector<EdgeW> sorted;
    while(true)
    {
        dag.transitive_reduction();
        sorted.clear();
        get_sorted_edges(dag, sorted, ignores);

        if(sorted.size() == 0)
            break;//No more fusible edges left

        EdgeW &e = sorted[0];
        Vertex a = source(e, dag.bglW());
        Vertex b = target(e, dag.bglW());
        if(path_exist(a, b, dag.bglD(), false))
            dag.merge_vertices(a, b);
        else
            dag.merge_vertices(b, a);

        //Note: since we call transitive_reduction() in each iteration,
        //the merge will never introduce cyclic dependencies.
        assert(not cycles(dag.bglD()));
    }
}

}} //namespace bohrium::dag

#endif

