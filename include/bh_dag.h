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
#include <stdexcept>
#include <bh.h>

/* Writes the DOT file of a DAG where
 * each vertex is a bh_ir_kernel.
 *
 * @dag       The DAG to write (of type 'Graph')
 * @filename  The name of DOT file
 * @header    Header string for the graph
 */
template <typename Graph>
void bh_dag_pprint(Graph &dag, const char filename[], const char *header = "")
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
 * NB: a vertex in the 'dag' must have an instr_list() and
 *     an add_instr() method.
 *
 * @a   The first vertex
 * @b   The second vertex
 * @dag The DAG
 */
template <typename Vertex, typename Graph>
void bh_dag_merge_vertex(const Vertex &a, const Vertex &b, Graph &dag)
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

/* Transitive reduce the 'dag', i.e. remove all redundant edges
 *
 * Complexity: O(E*(E + V))
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

#endif

