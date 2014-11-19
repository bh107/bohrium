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
template <typename Graph>
void bh_dag_from_bhir(const bh_ir &bhir, Graph &dag)
{
    using namespace std;
    using namespace boost;
    typedef typename graph_traits<Graph>::vertex_descriptor Vertex;

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
 * NB: a vertex in the 'dag' must bundle with the bh_ir_kernel class
 *
 * Complexity: O(E + V)
 *
 * @kernels The kernel list
 * @dag     The output dag
 */
template <typename Graph>
void bh_dag_from_kernels(const std::vector<bh_ir_kernel> &kernels, Graph &dag)
{
    using namespace std;
    using namespace boost;
    typedef typename graph_traits<Graph>::vertex_descriptor Vertex;

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
 * NB: a vertex in the 'dag' must bundle with the bh_ir_kernel class
 *
 * Complexity: O(E + V)
 *
 * @dag     The dag
 * @kernels The kernel list output
 */
template <typename Graph>
void bh_dag_fill_kernels(const Graph &dag, std::vector<bh_ir_kernel> &kernels)
{
    using namespace std;
    using namespace boost;
    typedef typename graph_traits<Graph>::vertex_descriptor Vertex;

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
template <typename Graph>
bool bh_dag_cycles(const Graph &g)
{
    using namespace std;
    using namespace boost;
    typedef typename graph_traits<Graph>::vertex_descriptor Vertex;
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

/* Merge two vertices in the 'dag', which invalidates all existing
 * vertex and edge pointers (boost descriptors)
 * NB: a vertex in the 'dag' must bundle with the bh_ir_kernel class
 *
 * @a   The first vertex
 * @b   The second vertex
 * @dag The DAG
 */
template <typename Vertex, typename Graph>
void bh_dag_merge_vertices(const Vertex &a, const Vertex &b, Graph &dag)
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
    clear_vertex(b, dag);
    remove_vertex(b, dag);
}

/* Merge the vertices specified by a list of edges and write
 * the result to new_dag, which should be empty.
 * NB: a vertex in 'dag' and 'new_dag' must bundle with the
 *     bh_ir_kernel class
 *
 * @dag              The input DAG
 * @edges2merge      List of edges that specifies which
 *                   vertices to merge
 * @new_dag          The output DAG
 * @check_fusibility Whether to throw a runtime error when
 *                   vertices isn't fusible
 */
template <typename Graph, typename Edge>
void bh_dag_merge_vertices(const Graph &dag,
                           const std::vector<Edge> edges2merge,
                           Graph &new_dag,
                           bool check_fusibility=false)
{
    using namespace std;
    using namespace boost;
    typedef typename graph_traits<Graph>::vertex_descriptor Vertex;

    //We use two vertex maps:
    //  One mapping between vertices in the old dag in which a vertex
    //  maps to the vertex in should be merged with.
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
    BOOST_FOREACH(const Edge &e, edges2merge)
    {
        const Vertex src = source(e,dag);
        const Vertex dst = target(e,dag);

        if(old2old[dst] == dst)
            old2old[dst] = old2old[src];
        else
            old2old[old2old[dst]] = old2old[src];
    }
    //For all common vertices we make old2new point to a new vertex
    BOOST_FOREACH(const Vertex &v, vertices(dag))
    {
        if(old2old[v] == v)
            old2new[v] = add_vertex(dag[v], new_dag);
    }
    //All merged vertices now point to one of the new vertices
    BOOST_FOREACH(const Vertex &v, vertices(dag))
    {
        if(old2old[v] != v)
            old2new[v] = old2new[old2old[v]];
    }
    BOOST_FOREACH(const Vertex &v, vertices(dag))
    {
        //Do the merging of instructions
        if(old2old[v] != v)
        {
            BOOST_FOREACH(const bh_instruction &i, dag[v].instr_list())
            {
                if(check_fusibility && !new_dag[old2new[v]].fusible(i))
                    throw runtime_error("Vertex not fusible!");
                new_dag[old2new[v]].add_instr(i);
            }
        }
        //Add edges to the new dag.
        BOOST_FOREACH(const Vertex &adj, adjacent_vertices(v, dag))
        {
            if(old2new[v] != old2new[adj])
                add_edge(old2new[v], old2new[adj], new_dag);
        }
    }
}

/* Determines the cost of the DAG. NB: a vertex in the 'dag'
 * must bundle with the bh_ir_kernel class
 *
 * Complexity: O(E + V)
 *
 * @dag     The DAG
 * @return  The cost
 */
template <typename Graph>
uint64_t bh_dag_cost(const Graph &dag)
{
    using namespace std;
    using namespace boost;
    typedef typename graph_traits<Graph>::vertex_descriptor Vertex;

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
template <typename Vertex, typename Graph>
bool bh_dag_long_path_exist(const Vertex &a, const Vertex &b, const Graph &dag)
{
    using namespace std;
    using namespace boost;
    typedef typename graph_traits<Graph>::edge_descriptor Edge;

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
 * NB: invalidates all existing vertex and edge pointers.
 *
 * Complexity: O(E * (E + V))
 *
 * @a   The first vertex
 * @b   The second vertex
 * @dag The DAG
 */
template <typename Graph>
void bh_dag_transitive_reduction(Graph &dag)
{
    using namespace std;
    using namespace boost;
    typedef typename graph_traits<Graph>::edge_descriptor Edge;

    vector<Edge> removal;
    BOOST_FOREACH(Edge e, edges(dag))
    {
        if(bh_dag_long_path_exist(source(e,dag), target(e,dag), dag))
            removal.push_back(e);
    }
    BOOST_FOREACH(Edge &e, removal)
    {
        remove_edge(e, dag);
    }
}

/* Retrieves all horizontal edges, i.e independent edges with non-zero cost
 *
 * Complexity: O(E * V)
 *
 * @dag   The DAG
 * @edges Output list of horizontal edges (as Vertex pairs)
 */
template <typename Graph, typename Vertex>
void bh_dag_horizontal_edges(const Graph &dag, std::vector<std::pair<Vertex,Vertex> > &edges)
{
    using namespace std;
    using namespace boost;
    typedef typename graph_traits<Graph>::edge_descriptor Edge;

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

            if(dag[*v1].dependency_cost(dag[*v2]) > 0)
            {
                if(bh_dag_long_path_exist(*v1, *v2, dag))
                    continue;
                if(bh_dag_long_path_exist(*v2, *v1, dag))
                    continue;
                edges.push_back(make_pair(*v1,*v2));
            }
        }
    }
}

/* Writes the DOT file of a DAG
 * NB: a vertex in the 'dag' must bundle with the bh_ir_kernel class
 *
 * Complexity: O(E + V)
 *
 * @dag       The DAG to write
 * @filename  The name of DOT file
 */
template <typename Graph>
void bh_dag_pprint(const Graph &dag, const char filename[])
{
    using namespace std;
    using namespace boost;
    typedef typename graph_traits<Graph>::vertex_descriptor Vertex;
    typedef typename graph_traits<Graph>::edge_descriptor Edge;
    typedef pair<Vertex,Vertex> hEdge;

    //Lets add horizontal edges as nondirectional edges in the DAG
    Graph new_dag(dag);
    vector<hEdge> h_edges1;//h-edges as Vertix pairs
    set<Edge> h_edges2;//h-edges as Edges
    bh_dag_horizontal_edges(new_dag, h_edges1);
    BOOST_FOREACH(const hEdge &e, h_edges1)
    {
        h_edges2.insert(add_edge(e.first, e.second, new_dag).first);
    }

    //We define a graph and a kernel writer for graphviz
    struct graph_writer
    {
        const Graph &graph;
        graph_writer(const Graph &g) : graph(g) {};
        void operator()(std::ostream& out) const
        {
            out << "labelloc=\"t\";" << endl;
            out << "label=\"DAG with a total cost of " << bh_dag_cost(graph);
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
 * NB: a vertex in the 'dag' must bundle with the bh_ir_kernel class
 *
 * Complexity: O(E * (V + E))
 *
 * @dag The DAG to fuse
 */
template <typename Graph>
void bh_dag_fuse_gentle(Graph &dag)
{
    using namespace std;
    using namespace boost;
    typedef typename graph_traits<Graph>::vertex_descriptor Vertex;

    bool not_finished;
    do
    {
        not_finished = false;
        BOOST_FOREACH(const Vertex &v, vertices(dag))
        {
            const auto child = adjacent_vertices(v, dag);
            const auto parent = inv_adjacent_vertices(v, dag);

            //If target vertex is a leaf (has no children)
            if(distance(child.first, child.second) == 0)
            {
                //And only have one parent
                if(distance(parent.first, parent.second) != 1)
                    continue;

                //And is gentle fusible with that parent
                if(not dag[*parent.first].fusible_gently(dag[v]))
                    continue;

                //The target and parent may fuse
                bh_dag_merge_vertices(*parent.first, v, dag);
                not_finished = true;
                break;
            }
            //If target vertex is a root (has no parents)
            if(distance(parent.first, parent.second) == 0)
            {
                //And only have one child
                if(distance(child.first, child.second) != 1)
                    continue;

                //And is gentle fusible with that child
                if(not dag[*child.first].fusible_gently(dag[v]))
                    continue;

                //The target and the child may fuse
                bh_dag_merge_vertices(*child.first, v, dag);
                not_finished = true;
                break;
            }
        }
    }while(not_finished);
}

/* Fuse vertices in the graph greedily, which is a non-optimal
 * algorithm that fuses the most costly edges in the DAG first.
 * NB: invalidates all existing vertex and edge pointers.
 * NB: a vertex in the 'dag' must bundle with the bh_ir_kernel class
 *
 * Complexity: O(E^2)
 *
 * @dag The DAG to fuse
 */
template <typename Graph>
void bh_dag_fuse_greedy(Graph &dag)
{
    using namespace std;
    using namespace boost;
    typedef typename graph_traits<Graph>::edge_descriptor Edge;
    typedef typename graph_traits<Graph>::vertex_descriptor Vertex;

    set<Edge> ignored;
    while(ignored.size() < num_edges(dag))
    {
        //Lets find the currect best edge
        Edge best;
        int64_t best_cost = -1;
        BOOST_FOREACH(const Edge &e, edges(dag))
        {
            if(ignored.find(e) != ignored.end())
                continue;
            const Vertex src = source(e, dag);
            const Vertex dst = target(e, dag);
            const int64_t cost = dag[dst].dependency_cost(dag[src]);
            if(cost > best_cost)
            {
                best_cost = cost;
                best = e;
            }
        }
        if(best_cost == -1)
            break;

        Graph new_dag(dag);
        bh_dag_merge_vertices(source(best, dag), target(best, dag), new_dag);
        if(bh_dag_cycles(new_dag))
        {
            ignored.insert(best);
        }
        else
        {
            dag = new_dag;
            ignored.clear();
        }
    }
}

/* Fuse vertices in the graph topologically, which is a non-optimal
 * algorithm that fuses based on the instruction order.
 * NB: invalidates all existing vertex and edge pointers.
 * NB: a vertex in the 'dag' must bundle with the bh_ir_kernel class
 *
 * Complexity: O(n) where 'n' is number of instruction
 *
 * @instr_list The instruction list
 * @dag        The output DAG
 */
template <typename Graph>
void bh_dag_fuse_topological(const std::vector<bh_instruction> &instr_list, Graph &dag)
{
    using namespace std;
    using namespace boost;
    typedef typename graph_traits<Graph>::vertex_descriptor Vertex;
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

#endif

