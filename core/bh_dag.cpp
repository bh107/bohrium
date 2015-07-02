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
#include <boost/graph/connected_components.hpp>
#include <boost/graph/strong_components.hpp>
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

void GraphDW::add_from_subgraph(const set<Vertex> &sub_graph, const GraphDW &dag)
{
    const GraphD &d = dag.bglD();
    const GraphW &w = dag.bglW();

    //Build the vertices of 'this' graph
    map<Vertex,Vertex> dag2this;//Maps from vertices in 'dag' to vertices in 'this'
    BOOST_FOREACH(Vertex v, sub_graph)
    {
        dag2this[v] = boost::add_vertex(d[v], _bglD);
        boost::add_vertex(_bglW);
    }
    //Add all dependency edges that connects vertices within 'sub_graph'
    BOOST_FOREACH(EdgeD e, edges(d))
    {
        Vertex src = source(e, d);
        Vertex dst = target(e, d);
        if(sub_graph.find(src) != sub_graph.end() and
           sub_graph.find(dst) != sub_graph.end())
        {
            boost::add_edge(dag2this[src], dag2this[dst], _bglD);
        }
    }
    //Add all weight edges that connects vertices within 'sub_graph'
    BOOST_FOREACH(EdgeW e, edges(w))
    {
        Vertex src = source(e, w);
        Vertex dst = target(e, w);
        if(sub_graph.find(src) != sub_graph.end() and
           sub_graph.find(dst) != sub_graph.end())
        {
            boost::add_edge(dag2this[src], dag2this[dst], w[e], _bglW);
        }
    }
}

void GraphDW::merge_vertices(Vertex a, Vertex b, bool a_before_b)
{
    assert(_bglD[a].fusible(_bglD[b]));

    //Merge the two instruction lists
    if(a_before_b)
    {
        BOOST_FOREACH(uint64_t idx, _bglD[b].instr_indexes)
            _bglD[a].add_instr(idx);
    }
    else
    {
        BOOST_FOREACH(uint64_t idx, _bglD[a].instr_indexes)
            _bglD[b].add_instr(idx);
        _bglD[a] = _bglD[b];//Only 'a' survives
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
            EdgeW e = *it++;
            Vertex v1 = source(e, _bglW);
            Vertex v2 = target(e, _bglW);
            int64_t cost = _bglD[v1].merge_cost_savings(_bglD[v2]);
            if(cost >= 0)
                _bglW[e].value = cost;
            else
                remove_edge(e, _bglW);
        }
    }
    assert(dag_validate(*this));
}

void GraphDW::transitive_reduction()
{
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
    assert(dag_validate(*this,false));
}

//Help function to check if 'base' is accessed by 'kernel'
static bool base_in_kernel(const bh_ir &bhir, const bh_ir_kernel &kernel,
                           const bh_base *base)
{
    for(uint64_t instr_idx: kernel.instr_indexes)
    {
        const bh_instruction &instr = bhir.instr_list[instr_idx];
        for(int i=0; i < bh_operands(instr.opcode); ++i)
        {
            if(bh_is_constant(&instr.operand[i]))
                continue;
            if(instr.operand[i].base == base)
                return true;
        }
    }
    return false;
}


void from_bhir(bh_ir &bhir, GraphDW &dag)
{
    assert(num_vertices(dag.bglD()) == 0);

    if(bhir.kernel_list.size() != 0)
    {
        throw logic_error("The kernel_list is not empty!");
    }

    uint64_t idx=0;
    while(idx < bhir.instr_list.size())
    {
        //Start new kernel
        bh_ir_kernel kernel(bhir);
        kernel.add_instr(idx);

        //Add gentle fusible instructions to the kernel
        for(idx=idx+1; idx < bhir.instr_list.size(); ++idx)
        {
            const bh_instruction &instr = bhir.instr_list[idx];

            //The new instruction must be system, only access one array, and
            //be equal to at least one array already in 'kernel'
            if(bh_opcode_is_system(instr.opcode) and \
               bh_operands(instr.opcode) == 1 and \
               base_in_kernel(bhir, kernel, instr.operand[0].base))
            {
                kernel.add_instr(idx);
            }
            else
                break;
        }
        dag.add_vertex(kernel);
    }
    assert(dag_validate(dag));
}

void from_kernels(const std::vector<bh_ir_kernel> &kernels, GraphDW &dag)
{
    assert(num_vertices(dag.bglD()) == 0);

    BOOST_FOREACH(const bh_ir_kernel &kernel, kernels)
    {
        if(kernel.instr_indexes.size() > 0)
            dag.add_vertex(kernel);
    }
    assert(dag_validate(dag));
}

void fill_kernel_list(const GraphD &dag, std::vector<bh_ir_kernel> &kernel_list)
{
    vector<Vertex> topological_order;
    topological_sort(dag, back_inserter(topological_order));
    BOOST_REVERSE_FOREACH(const Vertex &v, topological_order)
    {
        if(dag[v].instr_indexes.size() > 0)
            kernel_list.push_back(dag[v]);
    }
}

void split(const GraphDW &dag, vector<GraphDW> &output)
{
    const GraphD &d = dag.bglD();

    //Let's build a component graph
    typedef adjacency_list<setS, vecS, bidirectionalS, set<Vertex> > GraphComp;
    GraphComp comp_graph;
    {
        vector<Vertex> vertex2component(num_vertices(d));
        uint64_t num = connected_components(dag.bglW(), &vertex2component[0]);
        comp_graph = GraphComp(num);
        BOOST_FOREACH(Vertex v, vertices(d))
        {
            comp_graph[vertex2component[v]].insert(v);
        }
        BOOST_FOREACH(EdgeD e, edges(d))
        {
            Vertex src = vertex2component[source(e, d)];
            Vertex dst = vertex2component[target(e, d)];
            if(src != dst)
                add_edge(src, dst, comp_graph);
        }
    }

 //Print found components
 /*
    {
        BOOST_FOREACH(Vertex comp, vertices(comp_graph))
        {
            cout << "Component " << comp << ": ";
            BOOST_FOREACH(Vertex v, comp_graph[comp])
            {
                cout << v << ", ";
            }
            cout << endl;
        }
    }
*/

    //Merge strongly connected components in the component graph
    {
        //We start by finding the strongly connected components
        vector<Vertex> comp2connected(num_vertices(comp_graph));
        uint64_t num = strong_components(comp_graph, &comp2connected[0]);
        vector<set<Vertex> > connected2comp;
        connected2comp.resize(num);
        BOOST_FOREACH(Vertex v, vertices(comp_graph))
        {
            connected2comp[comp2connected[v]].insert(v);
        }
        //And the we merge them
        BOOST_FOREACH(set<Vertex> &comps, connected2comp)
        {
            if(comps.size() > 1)
            {
                auto root = comps.begin();
                auto it=root;
                for(++it; it != comps.end(); ++it)
                {
                    comp_graph[*root].insert(comp_graph[*it].begin(), comp_graph[*it].end());
                    comp_graph[*it].clear();
                    BOOST_FOREACH(Vertex v, adjacent_vertices(*it, comp_graph))
                    {
                        if(v != *root)
                            add_edge(*root, v, comp_graph);
                    }
                    BOOST_FOREACH(Vertex v, inv_adjacent_vertices(*it, comp_graph))
                    {
                        if(v != *root)
                            add_edge(v, *root, comp_graph);
                    }
                    boost::clear_vertex(*it, comp_graph);
                }
            }
        }
    }
    //Let's split the 'dag' in topological order
    vector<Vertex> topological_order;
    topological_sort(comp_graph, back_inserter(topological_order));
    BOOST_REVERSE_FOREACH(Vertex v, topological_order)
    {
        if(comp_graph[v].size() > 0)
        {
            output.resize(output.size()+1);
            output[output.size()-1].add_from_subgraph(comp_graph[v], dag);
        }
    }
}

bool path_exist(Vertex a, Vertex b, const GraphD &dag, bool only_long_path)
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
        if(only_long_path)
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
            const std::vector<bh_index>& ishape = graph[v].get_input_shape();
            for (size_t i = 0; i < ishape.size(); ++i)
                out << (i?", ":"[") << ishape[i];
            out << "], ";
            const std::vector<bh_index>& oshape = graph[v].get_output_shape();
            for (size_t i = 0; i < oshape.size(); ++i)
                out << (i?", ":"[") << oshape[i];
            out << "]      scalar: " <<  (graph[v].is_scalar()?"true":"false");
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
            out << "Constants: \\l";
            for (const bh_constant& c: graph[v].get_constants())
            {
                bh_sprint_const(&c, buf);
                out << buf << "\\l";
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
            out << "Sync base-arrays: \\l";
            BOOST_FOREACH(const bh_base* i, graph[v].get_syncs())
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
            int64_t cost = graph[src].merge_cost_savings(graph[dst]);
            int64_t weight = -1;
            bool directed = true;
            auto it = wmap.find(make_pair(src,dst));
            if(it != wmap.end())
            {
                tie(weight,directed) = (*it).second;
                assert(cost == weight);
            }
            out << "[label=\" ";
            if(cost == -1)
                out << "N/A\" color=red";
            else if(weight != -1)
                out << weight << " bytes\"";
            else
                out << "\"";
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


bool dag_validate(const GraphDW &dag, bool transitivity_allowed)
{
    const GraphD &d = dag.bglD();
    const GraphW &w = dag.bglW();
    if(num_vertices(d) != num_vertices(w))
    {
        cerr << "blgD and bglW has not the same number of vertices!" << endl;
        goto fail;
    }

    //Check for instruction duplications and vanishings
    {
        //Lets find all instructions
        set<uint64_t> instr_indexes;
        BOOST_FOREACH(Vertex v, vertices(d))
        {
            BOOST_FOREACH(uint64_t v, d[v].instr_indexes)
            {
                if(instr_indexes.find(v) != instr_indexes.end())
                {
                    cout << "Instruction [" << v << "] is in multiple kernels!" << endl;
                    goto fail;
                }
                instr_indexes.insert(v);
            }
        }
    }
    //Check for cycles
    if(cycles(d))
        goto fail;
    //Check precedence constraints
    BOOST_FOREACH(Vertex v1, vertices(d))
    {
        BOOST_FOREACH(Vertex v2, vertices(d))
        {
            if(v1 != v2)
            {
                const int dep = d[v1].dependency(d[v2]);
                if(dep == 1)//'v1' depend on 'v2'
                {
                    if(not path_exist(v2, v1, d))
                    {
                        cout << "Precedence check: not path between " << v1 \
                             << " and " << v2 << endl;
                        goto fail;
                    }
                }
            }
        }
    }
    //Check for weight edge inconsistency
    BOOST_FOREACH(EdgeD e, edges(d))
    {
        Vertex src = source(e, d);
        Vertex dst = target(e, d);
        int64_t cost = d[src].merge_cost_savings(d[dst]);
        if(cost != -1)//Is fusible
        {
            EdgeW we;
            bool we_exist;
            tie(we, we_exist) = edge(src,dst,w);
            if(we_exist)
            {
                if(w[we].value != cost)
                {
                    cout << "Weight check: weight of edge " << we \
                         << " is inconsistent with merge_cost_savings()!" << endl;
                    goto fail;
                }
            }
            else
            {
                cout << "Weight check: vertex " << src << " and " << dst \
                     << " is fusible and has a dependency edge but "\
                        "has no weight edge!" << endl;
                goto fail;
            }
        }
    }
    //Check transitivity
    if(not transitivity_allowed)
    {
        BOOST_FOREACH(EdgeD e, edges(d))
        {
            Vertex src = source(e, d);
            Vertex dst = target(e, d);
            if(path_exist(src, dst, d, true))
            {
                cout << "Transitivity check: dependency edge " << e \
                     << " is redundant!" << endl;
                goto fail;
            }
        }
        BOOST_FOREACH(EdgeW e, edges(w))
        {
            Vertex src = source(e, d);
            Vertex dst = target(e, d);
            if(path_exist(src, dst, d, true))
            {
                cout << "Transitivity check: weight edge " << e \
                     << " is redundant!" << endl;
                goto fail;
            }
        }
    }
    return true;
fail:
    cout << "writing validate-fail.dot" << endl;
    pprint(dag, "validate-fail.dot");
    return false;
}

bool dag_validate(const bh_ir &bhir, const vector<GraphDW> &dags, bool transitivity_allowed)
{
    set<uint64_t> instr_indexes;
    BOOST_FOREACH(const GraphDW &dag, dags)
    {
        const GraphD &d = dag.bglD();
        if(not dag_validate(dag, transitivity_allowed))
            return false;
        BOOST_FOREACH(Vertex v, boost::vertices(d))
        {
            BOOST_FOREACH(uint64_t idx, d[v].instr_indexes)
            {
                if(instr_indexes.find(idx) != instr_indexes.end())
                {
                    cout << "Instruction [" << idx << "] is in multiple kernels!" << endl;
                    goto fail;
                }
                instr_indexes.insert(idx);
            }
        }
    }
    //And check if all instructions exist
    for(uint64_t idx=0; idx<bhir.instr_list.size(); ++idx)
    {
        if(instr_indexes.find(idx) == instr_indexes.end())
        {
            cout << "Instruction [" << idx << "] is missing!" << endl;
            goto fail;
        }
    }
    return true;
fail:
    int i=0;
    cerr << "Dumping dot files:" << endl;
    BOOST_FOREACH(const GraphDW &dag, dags)
    {
        char t[1024];
        sprintf(t, "validate-fail-%d.dot", i++);
        cerr << t << endl;
        pprint(dag, t);
    }
    return false;
}

void fuse_gently(GraphDW &dag)
{
    const GraphD &d = dag.bglD();
    set<Vertex> vs(vertices(d).first, vertices(d).second);
    while(not vs.empty())
    {
        //Lets find and merge a "single ending" vertex
        set<Vertex>::iterator it=vs.begin();
        for(; it != vs.end(); ++it)
        {
            if(in_degree(*it, d) == 0 and out_degree(*it, d) == 1)
            {
                auto adj = adjacent_vertices(*it, d);
                Vertex v = *adj.first;
                if(d[*it].input_and_output_subset_of(d[v]))
                {
                    if(d[*it].fusible(d[v]))
                        dag.merge_vertices(v, *it, false);
                    vs.erase(it);
                    break;
                }
            }
            if(in_degree(*it, d) == 1 and out_degree(*it, d) == 0)
            {
                auto adj = inv_adjacent_vertices(*it, d);
                Vertex v = *adj.first;
                if(d[*it].input_and_output_subset_of(d[v]))
                {
                    if(d[*it].fusible(d[v]))
                        dag.merge_vertices(v, *it, true);
                    vs.erase(it);
                    break;
                }
            }
        }
        if(it == vs.end())
            break;//No single ending vertex found

        //Note that 'vs' is maintained since the extracted vertex '*it'
        //is the only vertex removed by the merge
    }
    dag.remove_cleared_vertices();
    assert(dag_validate(dag));
}

void fuse_greedy(GraphDW &dag)
{
    const GraphD &d = dag.bglD();
    const GraphW &w = dag.bglW();
    while(num_edges(w) > 0)
    {
        //Lets find the greatest weight edge.
        EdgeW greatest = *edges(w).first;
        BOOST_FOREACH(EdgeW e, edges(w))
        {
            if(w[e].value > w[greatest].value)
                greatest = e;
        }
        Vertex v1 = source(greatest, w);
        Vertex v2 = target(greatest, w);
        //And either remove it, if it is transitive
        if(path_exist(v1, v2, d, true) or path_exist(v2, v1, d, true))
        {
            dag.remove_edges(v1, v2);
        }
        else//Or merge it away
        {
            if(d[v1].dependency(d[v2]) == 1)//'v1' depend on 'v2'
                dag.merge_vertices(v2, v1);
            else
                dag.merge_vertices(v1, v2);
        }
    }
    dag.remove_cleared_vertices();
    assert(dag_validate(dag));
}

}} //namespace bohrium::dag

