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
#include <bh_dag.h>
#include "bh_fuse.h"

using namespace std;
using namespace boost;

namespace bohrium {
namespace dag {

void from_bhir(bh_ir &bhir, GraphDW &dag)
{
    assert(num_vertices(dag.bglD()) == 0);

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

void from_kernels(const std::vector<bh_ir_kernel> &kernels, GraphDW &dag)
{
    assert(num_vertices(dag.bglD()) == 0);

    BOOST_FOREACH(const bh_ir_kernel &kernel, kernels)
    {
        if(kernel.instr_indexes.size() > 0)
            dag.add_vertex(kernel);
    }
}

void fill_kernel_list(const GraphD &dag, std::vector<bh_ir_kernel> &kernel_list)
{
    assert(kernel_list.size() == 0);

    vector<Vertex> topological_order;
    topological_sort(dag, back_inserter(topological_order));
    BOOST_REVERSE_FOREACH(const Vertex &v, topological_order)
    {
        if(dag[v].instr_indexes.size() > 0)
            kernel_list.push_back(dag[v]);
    }
}

bool path_exist(Vertex a, Vertex b, const GraphD &dag, bool long_path)
{
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

bool cycles(const GraphD &g)
{
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

uint64_t dag_cost(const GraphD &dag)
{
    uint64_t cost = 0;
    BOOST_FOREACH(const Vertex &v, vertices(dag))
    {
        cost += dag[v].cost();
    }
    return cost;
}

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

void pprint(const GraphDW &dag, const char filename[])
{
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
            out << "Shape: ";
            const std::vector<bh_index>& shape = graph[v].get_shape();
            for (size_t i = 0; i < shape.size(); ++i)
                out << (i?", ":"[") << shape[i];
            out << "]  ";
            out << "\\lSweeps: ";
            for (const std::pair<bh_intp, bh_int64> &sweep: graph[v].get_sweeps())
            {
                out << "(" << sweep.first << ", " << sweep.second << ")  ";
            }
            out << "\\lInput views: \\l";
            BOOST_FOREACH(const bh_view &i, graph[v].get_input_set())
            {
                bh_sprint_view(&i, buf);
                out << graph[v].get_view_id(i) << ":" << buf << "\\l";
            }
            out << "Output views: \\l";
            BOOST_FOREACH(const bh_view &i, graph[v].get_output_set())
            {
                bh_sprint_view(&i, buf);
                out << graph[v].get_view_id(i) << ":" << buf << "\\l";
            }
            out << "Parameters: \\l";
            for (const std::pair<size_t,bh_base*>& p: graph[v].get_parameters())
            {
                bh_sprint_base(p.second, buf);
                out << "[" << p.first << "]" << buf << "\\l";
            }
            out << "Temp base-arrays: \\l";
            BOOST_FOREACH(const bh_base* i, graph[v].get_temps())
            {
                bh_sprint_base(i, buf);
                out << buf << "\\l";
            }
            out << "Free base-arrays: \\l";
            BOOST_FOREACH(const bh_base* i, graph[v].get_frees())
            {
                bh_sprint_base(i, buf);
                out << buf << "\\l";
            }
            out << "Discard base-arrays: \\l";
            BOOST_FOREACH(const bh_base* i, graph[v].get_discards())
            {
                bh_sprint_base(i, buf);
                out << buf << "\\l";
            }
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

bool dag_validate(const GraphD &dag)
{
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

bool dependency_subset(const GraphD &dag, Vertex sub, Vertex super)
{

    //The sub-vertex should have equal or less in- and out-degree.
    if(in_degree(sub,dag) > in_degree(super,dag))
        return false;
    if(out_degree(sub,dag) > out_degree(super,dag))
        return false;

    //Check that all adjacent vertices of 'sub' is also adjacent to 'super'
    BOOST_FOREACH(Vertex v1, adjacent_vertices(sub,dag))
    {
        if(v1 == super)
            continue;
        bool found = false;
        BOOST_FOREACH(Vertex v2, adjacent_vertices(super,dag))
        {
            if(v2 == sub)
                continue;
            if(v1 == v2)
            {
                found = true;
                break;
            }
        }
        if(not found)
            return false;
    }
    //Check that all inverse adjacent vertices of 'sub' is also inverse
    //adjacent to 'super'
    BOOST_FOREACH(Vertex v1, inv_adjacent_vertices(sub,dag))
    {
        if(v1 == super)
            continue;
        bool found = false;
        BOOST_FOREACH(Vertex v2, inv_adjacent_vertices(super,dag))
        {
            if(v2 == sub)
                continue;
            if(v1 == v2)
            {
                found = true;
                break;
            }
        }
        if(not found)
            return false;
    }
    return true;
}

void fuse_gently(GraphDW &dag)
{
    const GraphD &d = dag.bglD();
    set<EdgeD> dep_edges(edges(d).first, edges(d).second);
    /*
    char t[1024];
    sprintf(t, "tmp-%ld.dot", num_vertices(d));
    cout << t << endl;
    pprint(dag, t);
    */

    while(not dep_edges.empty())
    {
        //Lets find a "single ending" edge
        set<EdgeD>::iterator it=dep_edges.begin();
        for(; it != dep_edges.end(); ++it)
        {
            Vertex src = source(*it, d);
            Vertex dst = target(*it, d);
            if(in_degree(src, d) == 0 and out_degree(src, d) == 1 \
               and d[src].input_and_output_subset_of(d[dst]))
            {
                break;
            }
            else if(in_degree(dst, d) == 1 and out_degree(dst, d) == 0 \
               and d[dst].input_and_output_subset_of(d[src]))
            {
                break;
            }
        }
        if(it == dep_edges.end())
            break;//No single ending edge found

        //Lets extract the single ending edge
        EdgeD e = *it;
        dep_edges.erase(it);
        Vertex src = source(e, d);
        Vertex dst = target(e, d);

        //And merge them if able
        if(d[src].fusible(d[dst]))
            dag.merge_vertices(src, dst);

        //Note that 'dep_edges' is maintained since the extracted edge 'e'
        //is the only edge removed by the merge
    }
    dag.remove_cleared_vertices();
}

void fuse_greedy(GraphDW &dag, const std::set<Vertex> *ignores)
{
    //Help function to find and sort the weight edges.
    struct
    {
        void operator()(const GraphDW &g, vector<EdgeW> &edge_list,
                        const set<Vertex> *ignores)
        {
            if(ignores == NULL)
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
                    if(ignores->find(source(e, g.bglW())) == ignores->end() and
                       ignores->find(target(e, g.bglW())) == ignores->end())
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

