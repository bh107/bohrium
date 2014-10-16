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
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <stdexcept>
#include <bh.h>

/* Creates a new DAG based on a bhir that consist of single
 * instruction kernels.
 * NB: a vertex in the 'dag' must bundle with the bh_ir_kernel class
 *
 * Complexity: O(E + V)
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
    BOOST_FOREACH(const bh_instruction &instr, bhir.instr_list)
    {
        bh_ir_kernel k;
        k.add_instr(instr);
        Vertex new_v = add_vertex(k, dag);

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


/* Writes the DOT file of a DAG
 * NB: a vertex in the 'dag' must bundle with the bh_ir_kernel class
 *
 * Complexity: O(E + V)
 *
 * @dag       The DAG to write
 * @filename  The name of DOT file
 * @header    Header string for the graph
 */
template <typename Graph>
void bh_dag_pprint(const Graph &dag, const char filename[], const char *header = "")
{
    using namespace std;
    using namespace boost;

    //We define a graph and a kernel writer for graphviz
    struct graph_writer
    {
        const char *header;
        graph_writer(const char *h) : header(h) {};
        void operator()(std::ostream& out) const
        {
            out << "graph [bgcolor=white, fontname=\"Courier New\"]" << endl;
            out << "node [shape=box color=black, fontname=\"Courier New\"]" << endl;
            out << header << endl;
        }
    };
    typedef typename graph_traits<Graph>::vertex_descriptor Vertex;
    struct kernel_writer
    {
        const Graph &graph;
        kernel_writer(const Graph &g) : graph(g) {};
        void operator()(std::ostream& out, const Vertex& v) const
        {
            char buf[1024*10];
            out << "[label=\"Kernel " << v << "\\n";
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
    ofstream file;
    file.open(filename);
    write_graphviz(file, dag, kernel_writer(dag), default_writer(), graph_writer(header));
    file.close();
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
    BOOST_FOREACH(const Vertex &k, vertices(dag))
    {
        BOOST_FOREACH(const bh_view &v, dag[k].input_list())
        {
            cost += bh_nelements_nbcast(&v) * bh_type_size(v.base->type);
        }
        BOOST_FOREACH(const bh_view &v, dag[k].output_list())
        {
            cost += bh_nelements_nbcast(&v) * bh_type_size(v.base->type);
        }
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

/* Fuse vertices in the graph that can be fused without
 * changing any future possible fusings
 * NB: invalidates all existing vertex and edge pointers.
 * NB: a vertex in the 'dag' must bundle with the bh_ir_kernel class
 *
 * Complexity: O(E^2 * (V + E))
 *
 * @dag The DAG to fuse
 */
template <typename Graph>
void bh_dag_fuse_gentle(Graph &dag)
{
    using namespace std;
    using namespace boost;
    typedef typename graph_traits<Graph>::edge_descriptor Edge;
    typedef typename graph_traits<Graph>::vertex_descriptor Vertex;

    bool not_finished;
    do
    {
        not_finished = false;
        BOOST_FOREACH(Edge e, edges(dag))
        {
            Vertex src = source(e, dag);
            Vertex dst = target(e, dag);
            /* NOTE: if we check for cycles after merging we
             *  don't need the following two adjacent checks
            {
                auto adj = adjacent_vertices(src, dag);
                if(distance(adj.first, adj.second) != 1)
                    continue;
            }
            {
                auto adj = inv_adjacent_vertices(dst, dag);
                if(distance(adj.first, adj.second) != 1)
                    continue;
            }
            */
            if(!dag[src].fusible_gently(dag[dst]))
                continue;

            Graph new_dag(dag);
            bh_dag_merge_vertices(src, dst, new_dag);
            if(bh_dag_cycles(new_dag))
                continue;

            //We merged successfully
            dag = new_dag;
            not_finished = true;
            break;
        }
    }while(not_finished);
}

#endif

