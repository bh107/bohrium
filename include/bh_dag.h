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

//The GraphInstr extend the bh_instruction with an
//topological order ID. That is, if the memory access
//of two instructions overlaps, the instruction with
//the lower order ID precedes the other instruction.
struct GraphInstr
{
    const bh_instruction *instr;
    uint64_t order;

    GraphInstr(){}
    GraphInstr(const bh_instruction *i, uint64_t o):instr(i),order(o) {}
};

/* The GraphKernel class encapsulates a kernel of instructions
 * much like the bh_ir_kernel class. The main different is the
 * use of GraphInstr instead bh_ir_kernel.
 */
class GraphKernel
{
public:
    //The list of Bohrium instructions in this kernel
    //Note that this is a vertor of pointers and not a
    //copy of the original instruction list.
    std::vector<GraphInstr> instr_list;

    //List of input and output to this kernel.
    //NB: system instruction (e.g. BH_DISCARD) is
    //never part of kernel input or output
    std::vector<bh_view> input_list;
    std::vector<bh_view> output_list;

    //Lets of temporary base-arrays in this kernel.
    std::vector<const bh_base*> temp_list;

    /* Add an instruction reference to the kernel
     *
     * @instr   The instruction to add
     * @return  The boolean answer
     */
    void add_instr(const GraphInstr &instr)
    {
        using namespace std;
        using namespace boost;
        if(instr.instr->opcode == BH_DISCARD)
        {
            const bh_base *base = instr.instr->operand[0].base;
            for(vector<bh_view>::iterator it=output_list.begin();
                it != output_list.end(); ++it)
            {
                if(base == it->base)
                {
                    temp_list.push_back(base);
                    output_list.erase(it);
                    break;
                }
            }
        }
        else if(instr.instr->opcode != BH_FREE)
        {
            {
                bool duplicates = false;
                const bh_view &v = instr.instr->operand[0];
                BOOST_FOREACH(const bh_view &i, output_list)
                {
                    if(bh_view_aligned(&v, &i))
                    {
                        duplicates = true;
                        break;
                    }
                }
                if(!duplicates)

                    output_list.push_back(v);
            }
            const int nop = bh_operands(instr.instr->opcode);
            for(int i=1; i<nop; ++i)
            {
                const bh_view &v = instr.instr->operand[i];
                if(bh_is_constant(&v))
                    continue;

                bool duplicates = false;
                BOOST_FOREACH(const bh_view &i, input_list)
                {
                    if(bh_view_aligned(&v, &i))
                    {
                        duplicates = true;
                        break;
                    }
                }
                if(duplicates)
                    continue;

                bool local_source = false;
                BOOST_FOREACH(const GraphInstr &i, instr_list)
                {
                    if(bh_view_aligned(&v, &i.instr->operand[0]))
                    {
                        local_source = true;
                        break;
                    }
                }
                if(!local_source)
                    input_list.push_back(v);
            }
        }
        instr_list.push_back(instr);
    };

    /* Determines whether the kernel fusible legal
     *
     * @return The boolean answer
     */
    bool fusible() const
    {
        BOOST_FOREACH(const GraphInstr &i1, instr_list)
        {
            BOOST_FOREACH(const GraphInstr &i2, instr_list)
            {
                if(i1.instr != i2.instr)
                    if(not bohrium::check_fusible(i1.instr, i2.instr))
                        return false;
            }
        }
        return true;
    }

    /* Determines whether it is legal to fuse with the instruction
     *
     * @instr  The instruction
     * @return The boolean answer
     */
    bool fusible(const GraphInstr &instr) const
    {
        BOOST_FOREACH(const GraphInstr &i, instr_list)
        {
            if(not bohrium::check_fusible(i.instr, instr.instr))
                return false;
        }
        return true;
    }

    /* Determines whether it is legal to fuse with the kernel
     *
     * @other The other kernel
     * @return The boolean answer
     */
    bool fusible(const GraphKernel &other) const
    {
        BOOST_FOREACH(const GraphInstr &a, this->instr_list)
        {
            BOOST_FOREACH(const GraphInstr &b, other.instr_list)
            {
                if(not bohrium::check_fusible(a.instr, b.instr))
                    return false;
            }
        }
        return true;
    }

    /* Determines whether it is legal to fuse with the instruction
     * without changing this kernel's dependencies.
     *
     * @instr  The instruction
     * @return The boolean answer
     */
    bool fusible_gently(const GraphInstr &instr) const
    {
        if(bh_opcode_is_system(instr.instr->opcode))
            return true;

        //We are fusible if all instructions in this kernel are system opcodes
        {
            bool all_system = true;
            BOOST_FOREACH(const GraphInstr &i, instr_list)
            {
                if(not bh_opcode_is_system(i.instr->opcode))
                {
                    all_system = false;
                    break;
                }
            }
            if(all_system)
                return true;
        }
        //Check that 'instr' is fusible with least one existing instruction
        BOOST_FOREACH(const GraphInstr &i, instr_list)
        {
            if(bh_opcode_is_system(i.instr->opcode))
                continue;

            if(bh_instr_fusible_gently(instr.instr, i.instr) &&
               bohrium::check_fusible(instr.instr, i.instr))
                return true;
        }
        return false;
    }

    /* Determines whether it is legal to fuse with the kernel without
     * changing this kernel's dependencies.
     *
     * @other  The other kernel
     * @return The boolean answer
     */
    bool fusible_gently(const GraphKernel &other) const
    {

        BOOST_FOREACH(const GraphInstr &i, other.instr_list)
        {
            if(not fusible_gently(i))
                return false;
        }
        return true;
    }

    /* Determines whether this kernel depends on 'other',
     * which is true when:
     *      'other' writes to an array that 'this' access
     *                        or
     *      'this' writes to an array that 'other' access
     *
     * @other The other kernel
     * @return The boolean answer
     */
    bool dependency(const GraphKernel &other) const
    {
        BOOST_FOREACH(const GraphInstr &i, this->instr_list)
        {
            BOOST_FOREACH(const GraphInstr &o, other.instr_list)
            {
                if(bh_instr_dependency(i.instr, o.instr))
                    return true;
            }
        }
        return false;
    }

    /* Returns the cost of this kernel's dependency on the 'other' kernel.
     * The cost of a dependency is defined as the amount the BhIR will drop
     * in price if the two kernels are fused.
     * Note that a zero cost dependency is possible because of system
     * instructions such as BH_FREE and BH_DISCARD.
     *
     * @other  The other kernel
     * @return The cost value. Returns -1 if this and the 'other'
     *         kernel isn't fusible.
     */
    int64_t dependency_cost(const GraphKernel &other) const
    {
        if(this == &other)
            return 0;

        if(not fusible(other))
            return -1;

        int64_t price_drop = 0;

        //Subtract inputs that comes from 'other' or is already an input in 'other'
        BOOST_FOREACH(const bh_view &i, this->input_list)
        {
            BOOST_FOREACH(const bh_view &o, other.output_list)
            {
                if(bh_view_aligned(&i, &o))
                    price_drop += cost_of_view(i);
            }
            BOOST_FOREACH(const bh_view &o, other.input_list)
            {
                if(bh_view_aligned(&i, &o))
                    price_drop += cost_of_view(i);
            }
        }
        //Subtract outputs that are discared in 'this'
        BOOST_FOREACH(const bh_view &o, other.output_list)
        {
            BOOST_FOREACH(const GraphInstr &i, this->instr_list)
            {
                if(i.instr->opcode == BH_DISCARD and i.instr->operand[0].base == o.base)
                {
                    price_drop += cost_of_view(o);
                    break;
                }
            }
        }
        return price_drop;
    }

    /* Returns the cost of the kernel */
    uint64_t cost() const
    {
        uint64_t sum = 0;
        BOOST_FOREACH(const bh_view &v, input_list)
        {
            sum += cost_of_view(v);
        }
        BOOST_FOREACH(const bh_view &v, output_list)
        {
            sum += cost_of_view(v);
        }
        return sum;
    }
};

//The type declaration of the boost graphs, vertices and edges.
typedef boost::adjacency_list<boost::setS, boost::vecS, boost::bidirectionalS,
                              GraphKernel> GraphD;
typedef boost::adjacency_list<boost::setS, boost::vecS, boost::undirectedS,
                              boost::no_property, EdgeWeight> GraphW;
typedef typename boost::graph_traits<GraphD>::edge_descriptor EdgeD;
typedef typename boost::graph_traits<GraphW>::edge_descriptor EdgeW;

//Forward declaration
bool path_exist(Vertex a, Vertex b, const GraphD &dag,
                bool ignore_neighbors);

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
    Vertex add_vertex(const GraphKernel &kernel)
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
                if(kernel.dependency(_bglD[v]))
                {
                    dependency = true;
                    boost::add_edge(v, d, _bglD);
                }
                int64_t cost = kernel.dependency_cost(_bglD[v]);
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
        _bglD[v] = GraphKernel();
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
            if(_bglD[v].instr_list.size() == 0)
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

        BOOST_FOREACH(const GraphInstr &i, _bglD[b].instr_list)
        {
            _bglD[a].add_instr(i);
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
                int64_t cost = _bglD[v2].dependency_cost(_bglD[v1]);
                if(cost >= 0)
                    _bglW[e].value = cost;
                else
                    edges2remove.push_back(e);
            }
            else if(path_exist(v2, v1, _bglD, false))
            {
                int64_t cost = _bglD[v1].dependency_cost(_bglD[v2]);
                if(cost >= 0)
                    _bglW[e].value = cost;
                else
                    edges2remove.push_back(e);
            }
            else
            {
                int64_t cost = _bglD[v1].dependency_cost(_bglD[v2]);
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
                assert( 1 == 2);
            }
        }
        if(not _bglD[a].fusible())
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
        BOOST_FOREACH(EdgeD e, edges(_bglD))
        {
            if(path_exist(source(e,_bglD), target(e,_bglD), _bglD, true))
               boost::remove_edge(e, _bglD);
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
void from_bhir(const bh_ir &bhir, GraphDW &dag)
{
    using namespace std;
    using namespace boost;

    if(bhir.kernel_list.size() != 0)
    {
        throw logic_error("The kernel_list is not empty!");
    }
    //Build a singleton DAG
    uint64_t count = 0;
    BOOST_FOREACH(const bh_instruction &instr, bhir.instr_list)
    {
        GraphKernel k;
        k.add_instr(GraphInstr(&instr, count++));
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

    uint64_t count = 0;
    BOOST_FOREACH(const bh_ir_kernel &kernel, kernels)
    {
        if(kernel.instr_list().size() == 0)
            continue;

        GraphKernel k;
        BOOST_FOREACH(const bh_instruction &instr, kernel.instr_list())
        {
            k.add_instr(GraphInstr(&instr, count++));
        }
        dag.add_vertex(k);
    }
}

/* Fills the kernel list based on the DAG where each vertex is a kernel.
 *
 * Complexity: O(E + V)
 *
 * @dag     The dag
 * @kernels The kernel list output
 */
void fill_kernels(const GraphD &dag, std::vector<bh_ir_kernel> &kernels)
{
    using namespace std;
    using namespace boost;

    vector<Vertex> topological_order;
    topological_sort(dag, back_inserter(topological_order));
    BOOST_REVERSE_FOREACH(const Vertex &v, topological_order)
    {
        if(dag[v].instr_list.size() > 0)
        {
            const GraphKernel &gk = dag[v];
            bh_ir_kernel k;

            k.inputs = gk.input_list;
            k.outputs = gk.output_list;
            k.temps = gk.temp_list;

            k.instrs.reserve(gk.instr_list.size());
            BOOST_FOREACH(const GraphInstr &i, gk.instr_list)
            {
                k.instrs.push_back(*i.instr);
            }
            kernels.push_back(k);
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
            BOOST_FOREACH(const bh_view &i, graph[v].input_list)
            {
                bh_sprint_view(&i, buf);
                out << buf << "\\l";
            }
            out << "Output views: \\l";
            BOOST_FOREACH(const bh_view &i, graph[v].output_list)
            {
                bh_sprint_view(&i, buf);
                out << buf << "\\l";
            }
            out << "Temp base-arrays: \\l";
            BOOST_FOREACH(const bh_base *i, graph[v].temp_list)
            {
                bh_sprint_base(i, buf);
                out << buf << "\\l";
            }
            out << "Instruction list: \\l";
            BOOST_FOREACH(const GraphInstr &i, graph[v].instr_list)
            {
                out << "[" << i.order << "] ";
                bh_sprint_instr(i.instr, buf, "\\l");
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

    bool not_finished = true;
    vector<EdgeW> sorted;
    while(not_finished)
    {
        not_finished = false;
        sorted.clear();
        get_sorted_edges(dag, sorted, ignores);
        BOOST_FOREACH(const EdgeW &e, sorted)
        {
            GraphDW new_dag(dag);
            Vertex a = source(e, dag.bglW());
            Vertex b = target(e, dag.bglW());
            if(path_exist(a, b, dag.bglD(), false))
                new_dag.merge_vertices(a, b);
            else
                new_dag.merge_vertices(b, a);

            if(not cycles(new_dag.bglD()))
            {
                dag = new_dag;
                not_finished = true;
                break;
            }
        }
    }
}

}} //namespace bohrium::dag

#endif

