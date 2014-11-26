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

namespace bohrium {
namespace dag {

typedef boost::adjacency_list<boost::setS, boost::vecS, boost::bidirectionalS, bh_ir_kernel> Graph;
typedef typename boost::graph_traits<Graph>::vertex_descriptor Vertex;
typedef typename boost::graph_traits<Graph>::edge_descriptor Edge;

/* Class that represents a weight between two vertices */
class EdgeW
{
public:
    //The topological sorted vertex pair
    std::pair<Vertex,Vertex> edge;
    //The weight between the vertex pair
    uint64_t weight;

    EdgeW(Vertex v1, Vertex v2, uint64_t w=0)
    {
        edge.first = v1;
        edge.second = v2;
        weight = w;
    }
};

/* Creates a new DAG based on a bhir that consist of single
 * instruction kernels.
 * NB: a vertex in the 'dag' must bundle with the bh_ir_kernel class
 *
 * Complexity: O(n^2) where 'n' is the number of instructions
 *
 * @bhir  The BhIR
 * @dag   The output dag
 *
 * Throw logic_error() if the kernel_list wihtin 'bhir' isn't empty
 */
void from_bhir(const bh_ir &bhir, Graph &dag)
{
    using namespace std;
    using namespace boost;

    if(bhir.kernel_list.size() != 0)
    {
        throw logic_error("The kernel_list is not empty!");
    }
    //Build a singleton DAG
    BOOST_FOREACH(const bh_instruction &instr, bhir.instr_list)
    {
        Vertex new_v = add_vertex(dag);
        bh_ir_kernel &k = dag[new_v];
        k.add_instr(instr);

        //Add dependencies
        BOOST_FOREACH(Vertex v, vertices(dag))
        {
            if(new_v != v)//We do not depend on ourself
            {
                if(k.dependency(dag[v]))
                    add_edge(v, new_v, dag);
            }
        }
    }
}

/* Creates a new DAG based on a kernel list where each vertex is a kernel.
 *
 * Complexity: O(E + V)
 *
 * @kernels The kernel list
 * @dag     The output dag
 */
void from_kernels(const std::vector<bh_ir_kernel> &kernels, Graph &dag)
{
    using namespace std;
    using namespace boost;

    BOOST_FOREACH(const bh_ir_kernel &kernel, kernels)
    {
        if(kernel.instr_list().size() == 0)
            continue;

        Vertex new_v = add_vertex(kernel, dag);

        //Add dependencies
        BOOST_FOREACH(Vertex v, vertices(dag))
        {
            if(new_v != v)//We do not depend on ourself
            {
                if(kernel.dependency(dag[v]))
                    add_edge(v, new_v, dag);
            }
        }
    }
}

/* Fills the kernel list based on the DAG where each vertex is a kernel.
 *
 * Complexity: O(E + V)
 *
 * @dag     The dag
 * @kernels The kernel list output
 */
void fill_kernels(const Graph &dag, std::vector<bh_ir_kernel> &kernels)
{
    using namespace std;
    using namespace boost;

    vector<Vertex> topological_order;
    topological_sort(dag, back_inserter(topological_order));
    BOOST_REVERSE_FOREACH(const Vertex &v, topological_order)
    {
        if(dag[v].instr_list().size() > 0)
            kernels.push_back(dag[v]);
    }
}

/* Determines whether there are cycles in the Graph
 *
 * Complexity: O(E + V)
 *
 * @g       The digraph
 * @ba      The second vertex
 * @other   The other kernel
 * @return  True if there are cycles in the digraph, else false
 */
bool cycles(const Graph &g)
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

/* Clear the vertex without actually removing it.
 * NB: invalidates all existing edge iterators
 *     but NOT pointers to neither vertices nor edges.
 *
 * Complexity: O(1)
 *
 * @dag  The DAG
 * @v    The Vertex
 */
void nullify_vertex(Graph &dag, const Vertex &v)
{
    boost::clear_vertex(v, dag);
    dag[v] = bh_ir_kernel();
}

/* Merge vertex 'a' and 'b' by appending 'b's instructions to 'a'.
 * Vertex 'b' is nullified rather than removed thus existing vertex
 * and edge pointers are still valid after the merge.
 *
 * NB: invalidates all existing edge iterators.
 *
 * Complexity: O(1)
 *
 * @a   The first vertex
 * @b   The second vertex
 * @dag The DAG
 */
void merge_vertices(const Vertex &a, const Vertex &b, Graph &dag)
{
    using namespace std;
    using namespace boost;

    BOOST_FOREACH(const bh_instruction &i, dag[b].instr_list())
    {
        dag[a].add_instr(i);
    }
    BOOST_FOREACH(const Vertex &v, adjacent_vertices(b, dag))
    {
        if(a != v)
            add_edge(a, v, dag);
    }
    BOOST_FOREACH(const Vertex &v, inv_adjacent_vertices(b, dag))
    {
        if(a != v)
            add_edge(v, a, dag);
    }
    nullify_vertex(dag, b);
}

/* Merge the vertices specified by a list of edges and write
 * the result to new_dag, which should be empty.
 *
 * Complexity: O(V + E)
 *
 * @dag              The input DAG
 * @edges2merge      List of weight edges that specifies which
 *                   pair of vertices to merge
 * @new_dag          The output DAG
 * @check_fusibility Whether to throw a runtime error when
 *                   vertices isn't fusible
 */
void merge_vertices(const Graph &dag,
                    const std::vector<EdgeW> edges2merge,
                    Graph &new_dag,
                    bool check_fusibility = false)
{
    using namespace std;
    using namespace boost;

    //Help function to find the common vertex
    struct find_common_vertex_in_old2old_map
    {
        Vertex find(map<Vertex, Vertex> &old2old, Vertex original, Vertex v)
        {
            Vertex v_mapped = old2old[v];
            if(v_mapped != v)
                return this->find(old2old, original, v_mapped);
            else
                return v;
        }
        Vertex operator()(map<Vertex, Vertex> &old2old, Vertex v)
        {
            Vertex v_mapped = old2old[v];
            if(v_mapped == v)
                return v;
            else
                return this->find(old2old, v, v_mapped);
        }
    }find_common;

    //We use two vertex maps:
    //  One mapping between vertices in the old dag in which a vertex
    //  maps to the vertex it should be merged with.
    map<Vertex, Vertex> old2old;
    //  Another mapping from vertices in the old dag to vertices in the new dag.
    map<Vertex, Vertex> old2new;
    //Initially old2old is a simple identity map
    BOOST_FOREACH(const Vertex &v, vertices(dag))
    {
        old2old[v] = v;
    }
    //Then we make merged vertices point to a common vertex
    //(a vertex where old2old[v] == v holds). Note that old2old
    //is now a surjective map.
    BOOST_FOREACH(const EdgeW &e, edges2merge)
    {
        const Vertex &v1 = e.edge.first;
        const Vertex &v2 = e.edge.second;
        //We find the common vertex of 'v1' and makes it point to the common vertex of 'v2'
        old2old[find_common(old2old, v1)] = find_common(old2old, v2);
    }
    //For all common vertices we make old2new point to a new empty vertex
    BOOST_FOREACH(const Vertex &v, vertices(dag))
    {
        if(old2old[v] == v)
            old2new[v] = add_vertex(new_dag);
    }
    //All merged vertices now point to one of the new vertices
    BOOST_FOREACH(const Vertex &v, vertices(dag))
    {
        if(old2old[v] != v)
            old2new[v] = old2new[find_common(old2old, v)];
    }

    //Finally we merge the instruction into the their common vertices topologically
    vector<Vertex> topological_order;
    topological_sort(dag, back_inserter(topological_order));
    BOOST_REVERSE_FOREACH(const Vertex &v, topological_order)
    {
        //Do the merging of instructions
        BOOST_FOREACH(const bh_instruction &i, dag[v].instr_list())
        {
            if(check_fusibility && !new_dag[old2new[v]].fusible(i))
                throw runtime_error("Vertex not fusible!");
            new_dag[old2new[v]].add_instr(i);
        }
        //Add edges to the new dag.
        BOOST_FOREACH(const Vertex &adj, adjacent_vertices(v, dag))
        {
            if(old2new[v] != old2new[adj])
                add_edge(old2new[v], old2new[adj], new_dag);
        }
    }
}

/* Determines the cost of the DAG.
 *
 * Complexity: O(E + V)
 *
 * @dag     The DAG
 * @return  The cost
 */
uint64_t dag_cost(const Graph &dag)
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

/* Determines whether there exist a path from 'a' to 'b' with
 * length more than one ('a' and 'b' is not adjacent).
 *
 * Complexity: O(E + V)
 *
 * @a       The first vertex
 * @b       The second vertex
 * @dag     The DAG
 * @return  True if there is a long path, else false
 */
bool long_path_exist(const Vertex &a, const Vertex &b, const Graph &dag)
{
    using namespace std;
    using namespace boost;

    struct local_visitor:default_bfs_visitor
    {
        const Vertex src, dst;
        local_visitor(const Vertex &a, const Vertex &b):src(a),dst(b){};

        void examine_edge(Edge e, const Graph &g) const
        {
            if(source(e,g) != src)
            {
                if(target(e,g) == dst)
                {
                    throw runtime_error("");
                }
            }
        }
    };
    try
    {
        breadth_first_search(dag, a, visitor(local_visitor(a,b)));
    }
    catch (const runtime_error &e)
    {
        return true;
    }
    return false;
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
void transitive_reduction(Graph &dag)
{
    using namespace std;
    using namespace boost;

    vector<Edge> removal;
    BOOST_FOREACH(Edge e, edges(dag))
    {
        if(long_path_exist(source(e,dag), target(e,dag), dag))
            removal.push_back(e);
    }
    BOOST_FOREACH(Edge &e, removal)
    {
        remove_edge(e, dag);
    }
}

/* Retrieves all horizontal edges, i.e independent edges with non-zero cost
 *
 * Complexity: O(V^2 * E)
 *
 * @dag   The DAG
 * @edges Output list of horizontal edges
 */
void horizontal_edges(const Graph &dag, std::vector<EdgeW> &edges)
{
    using namespace std;
    using namespace boost;

    typename graph_traits<Graph>::vertex_iterator v1, v2, v_end;
    for(tie(v1, v_end) = vertices(dag); v1 != v_end; ++v1)
    {
        for(v2=v1+1; v2 != v_end; ++v2)
        {
            Edge e;
            bool exist;
            tie(e, exist) = edge(*v1, *v2, dag);
            if(exist)
                continue;
            tie(e, exist) = edge(*v2, *v1, dag);
            if(exist)
                continue;
            int64_t w = dag[*v1].dependency_cost(dag[*v2]);
            if(w > 0)
            {
                if(long_path_exist(*v1, *v2, dag))
                    continue;
                if(long_path_exist(*v2, *v1, dag))
                    continue;
                edges.push_back(EdgeW(*v1,*v2, w));
            }
        }
    }
}

/* Retrieve all vertical and horizontal weights in a DAG
 *
 * Complexity: O(V^2 * E)
 *
 * @dag       The DAG
 * @edges  The output edge list
 */
void all_weights(const Graph &dag, std::vector<EdgeW> &edges)
{
    horizontal_edges(dag, edges);
    BOOST_FOREACH(const Edge &e, boost::edges(dag))
    {
        Vertex src = source(e,dag);
        Vertex dst = target(e,dag);
        int64_t w = dag[dst].dependency_cost(dag[src]);
        if(w >= 0)
        {
            edges.push_back(EdgeW(src, dst, w));
        }
    }
}

/* Sort the weights in descending order
 *
 * Complexity: O(E * log E)
 *
 * @edges  The input/output edge list
 */
void sort_weights(std::vector<EdgeW> &edges)
{
    struct wcmp
    {
        bool operator() (const EdgeW &e1, const EdgeW &e2)
        {
            return (e1.weight > e2.weight);
        }
    };
    sort(edges.begin(), edges.end(), wcmp());
}

/* Writes the DOT file of a DAG
 *
 * Complexity: O(E + V)
 *
 * @dag       The DAG to write
 * @filename  The name of DOT file
 */
void pprint(const Graph &dag, const char filename[])
{
    using namespace std;
    using namespace boost;
    //Lets add horizontal edges as nondirectional edges in the DAG
    Graph new_dag(dag);
    vector<EdgeW> h_edges1;//h-edges as EdgeWs
    set<Edge> h_edges2;//h-edges as Edges
    horizontal_edges(new_dag, h_edges1);
    BOOST_FOREACH(const EdgeW &e, h_edges1)
    {
        h_edges2.insert(add_edge(e.edge.first, e.edge.second, new_dag).first);
    }

    //We define a graph and a kernel writer for graphviz
    struct graph_writer
    {
        const Graph &graph;
        graph_writer(const Graph &g) : graph(g) {};
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
        const Graph &graph;
        kernel_writer(const Graph &g) : graph(g) {};
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
            BOOST_FOREACH(const bh_instruction &i, graph[v].instr_list())
            {
                bh_sprint_instr(&i, buf, "\\l");
                out << buf << "\\l";
            }
            out << "\"]";
        }
    };
    struct edge_writer
    {
        const Graph &graph;
        const set<Edge> &h_edges;
        edge_writer(const Graph &g, const set<Edge> &e) : graph(g), h_edges(e) {};
        void operator()(std::ostream& out, const Edge& e) const
        {
            int64_t c = graph[target(e,graph)].dependency_cost(graph[source(e,graph)]);
            out << "[label=\" ";
            if(c == -1)
                out << "N/A\" color=red";
            else
                out << c << " bytes\"";
            if(h_edges.find(e) != h_edges.end())
                out << " dir=none color=green constraint=false";
            out << "]";
        }
    };
    ofstream file;
    file.open(filename);
    write_graphviz(file, new_dag, kernel_writer(new_dag),
                   edge_writer(new_dag, h_edges2), graph_writer(new_dag));
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
void fuse_gentle(Graph &dag)
{
    using namespace std;
    using namespace boost;

    vector<EdgeW> edges2merge;
    do
    {
        edges2merge.clear();
        BOOST_FOREACH(const Edge &e, edges(dag))
        {
            const Vertex &src = source(e, dag);
            const Vertex &dst = target(e, dag);
            if((in_degree(dst, dag) == 1 and out_degree(dst, dag) == 0) or
               (in_degree(src, dag) == 0 and out_degree(src, dag) == 1) or
               (in_degree(dst, dag) <= 1 and out_degree(src, dag) <= 1))
            {
                if(dag[dst].fusible_gently(dag[src]))
                {
                    edges2merge.push_back(EdgeW(src, dst));
                }
            }
        }
        Graph new_dag;
        merge_vertices(dag, edges2merge, new_dag);
        dag = new_dag;
    }
    while(edges2merge.size() > 0);
}

/* Fuse vertices in the graph greedily, which is a non-optimal
 * algorithm that fuses the most costly edges in the DAG first.
 * NB: invalidates all existing edge iterators.
 *
 * Complexity: O(E^2 * (E + V))
 *
 * @dag The DAG to fuse
 */
void fuse_greedy(Graph &dag)
{
    using namespace std;
    using namespace boost;

    bool not_finished = true;
    while(not_finished)
    {
        vector<EdgeW> edge_list;
        all_weights(dag, edge_list);
        sort_weights(edge_list);
        not_finished = false;
        BOOST_FOREACH(const EdgeW &e, edge_list)
        {
            Graph new_dag(dag);
            merge_vertices(e.edge.first, e.edge.second, new_dag);
            if(not cycles(new_dag))
            {
                dag = new_dag;
                not_finished = true;
                break;
            }
        }
    }
}

/* Fuse vertices in the graph topologically, which is a non-optimal
 * algorithm that fuses based on the instruction order.
 *
 * Complexity: O(n) where 'n' is number of instruction
 *
 * @instr_list The instruction list
 * @dag        The output DAG
 */
void fuse_topological(const std::vector<bh_instruction> &instr_list, Graph &dag)
{
    using namespace std;
    using namespace boost;
    Vertex vNULL = graph_traits<Graph>::null_vertex();

    //Find kernels
    vector<bh_ir_kernel> kernel_list;
    vector<bh_instruction>::const_iterator it = instr_list.begin();
    while(it != instr_list.end())
    {
        bh_ir_kernel kernel;
        kernel.add_instr(*it);

        int i=1;
        for(it=it+1; it != instr_list.end(); ++it, ++i)
        {
            if(kernel.fusible(*it))
            {
                kernel.add_instr(*it);
            }
            else
                break;
        }
        kernel_list.push_back(kernel);
    }

    //Fill the DAG
    uint64_t i=0;
    Vertex prev = vNULL;
    dag = Graph(kernel_list.size());
    BOOST_FOREACH(const Vertex &v, vertices(dag))
    {
        dag[v] = kernel_list[i++];
        if(prev != vNULL)
            add_edge(prev, v, dag);
        prev = v;
    }
}

}} //namespace bohrium::dag

#endif

