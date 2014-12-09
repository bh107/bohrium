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
#include <bh.h>
#include <bh_dag.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <boost/foreach.hpp>
#include <boost/graph/topological_sort.hpp>
#include <vector>
#include <map>
#include <iterator>

#define VERBOSE

using namespace std;
using namespace boost;
using namespace bohrium::dag;

pair<int64_t,bool> fuse_mask(int64_t best_cost, const vector<EdgeW> &edges2explore,
                             const GraphDW graph, const vector<bool> &mask, GraphD &dag)
{
    bool fusibility=true;
    vector<EdgeW> edges2merge;
    unsigned int i=0;
    BOOST_FOREACH(const EdgeW &e, edges2explore)
    {
        if(mask[i++])
        {
            edges2merge.push_back(e);
        }
    }

    //Help function to find the new location
    struct find_new_location
    {
        Vertex operator()(map<Vertex, Vertex> &loc_map, Vertex v)
        {
            Vertex v_mapped = loc_map[v];
            if(v_mapped == v)
                return v;
            else
                return (*this)(loc_map, v_mapped);
        }
    }find_loc;

    //'loc_map' maps a vertex before the merge to the corresponding vertex after the merge
    map<Vertex, Vertex> loc_map;
    BOOST_FOREACH(Vertex v, vertices(dag))
    {
        loc_map[v] = v;
    }

    BOOST_FOREACH(const EdgeW &e, edges2merge)
    {
        Vertex v1 = find_loc(loc_map, source(e, graph.bglW()));
        Vertex v2 = find_loc(loc_map, target(e, graph.bglW()));
        loc_map[v1] = v2;
    }

    map<Vertex, bh_ir_kernel> new_vertices;
    BOOST_FOREACH(Vertex v, vertices(dag))
    {
        if(loc_map[v] == v)
            new_vertices[v] = bh_ir_kernel();
    }

    vector<Vertex> topological_order;
    topological_sort(dag, back_inserter(topological_order));
    BOOST_REVERSE_FOREACH(Vertex vertex, topological_order)
    {
        Vertex v = find_loc(loc_map, vertex);
        bh_ir_kernel &k = new_vertices.at(v);
        BOOST_FOREACH(const bh_instruction &i, dag[vertex].instr_list())
        {
            if(not k.fusible(i))
                fusibility = false;
            k.add_instr(i);
        }
    }

    int64_t cost=0;
    BOOST_FOREACH(Vertex v, vertices(dag))
    {
        if(loc_map[v] == v)
            cost += new_vertices[v].cost();
    }

    if(cost >= best_cost or not fusibility)
        return make_pair(cost,fusibility);

    BOOST_FOREACH(Vertex v, vertices(dag))
    {
        Vertex loc_v = loc_map[v];
        if(loc_v == v)
        {
            dag[v] = new_vertices[v];
        }
        else//Lets merge 'v' into 'loc_v'
        {
            BOOST_FOREACH(Vertex a, adjacent_vertices(v, dag))
            {
                if(a != loc_v)
                    add_edge(loc_v, a, dag);
            }
            BOOST_FOREACH(Vertex a, inv_adjacent_vertices(v, dag))
            {
                if(a != loc_v)
                    add_edge(a, loc_v, dag);
            }
            clear_vertex(v, dag);
            dag[v] = bh_ir_kernel();
        }
    }
    if(cycles(dag))
    {
        return make_pair(cost,false);
    }

    if(cost != (int64_t)dag_cost(dag))
    {
        cout << "cost: " << cost << ", dag_cost: " << dag_cost(dag) << endl;
        GraphDW tmp(dag);
        pprint(tmp, "what.dot");
        assert(1 == 2);
    }

    return make_pair(cost,true);
}
#ifdef VERBOSE
double  purge_count=0;
uint64_t explore_count=0;
int fuser_count=0;
#endif
int64_t best_cost;
int64_t one_cost;
GraphD best_dag;
void fuse(const GraphDW &dag, const vector<EdgeW> &edges2explore,
          vector<bool> mask, unsigned int offset, bool merge_next)
{
    if(not merge_next)
    {
        GraphD new_dag(dag.bglD());
        mask[offset] = merge_next;
        bool fusibility;
        int64_t cost;
        tie(cost, fusibility) = fuse_mask(best_cost, edges2explore, dag, mask, new_dag);


#ifdef VERBOSE
        if(explore_count%1000 == 0)
        {
            cout << "[" << explore_count << "] " << "purge count: " << purge_count << " / " << pow(2.0,mask.size()) << endl;
            cout << "cost: " << cost << ", best_cost: " << best_cost << ", fusibility: " << fusibility << endl;
        }
        ++explore_count;
#endif

        if(cost >= best_cost)
        {
#ifdef VERBOSE
            purge_count += pow(2.0, mask.size()-offset-1);
#endif
            return;
        }
        if(fusibility)
        {
            best_cost = cost;
            best_dag = new_dag;
#ifdef VERBOSE
            std::stringstream ss;
            ss << "new_best_dag-" << fuser_count << "-" << dag_cost(new_dag) << ".dot";
            printf("write file: %s\n", ss.str().c_str());
            pprint(GraphDW(new_dag), ss.str().c_str());
            purge_count += pow(2.0, mask.size()-offset-1);
#endif
            return;
        }
    }
    if(offset+1 < mask.size())
    {
        fuse(dag, edges2explore, mask, offset+1, false);
        fuse(dag, edges2explore, mask, offset+1, true);
    }
}

void fuser(bh_ir &bhir)
{
#ifdef VERBOSE
    ++fuser_count;
#endif

    GraphDW dag;
    from_bhir(bhir, dag);
    fuse_gentle(dag);
    dag.transitive_reduction();

    //The list of edges that we should try to merge
    vector<EdgeW> edges2explore;
    BOOST_FOREACH(const EdgeW &e, edges(dag.bglW()))
    {
        edges2explore.push_back(e);
    }
    sort_weights(dag.bglW(), edges2explore);
    //reverse(edges2explore.begin(), edges2explore.end());

    if(edges2explore.size() == 0)
    {
        fill_kernels(dag.bglD(), bhir.kernel_list);
        return;
    }

    //First we check the trivial case where all kernels are merged
    vector<bool> mask(edges2explore.size(), true);
    {
        GraphD new_dag(dag.bglD());
        bool fuse;
        tie(one_cost, fuse) = fuse_mask(numeric_limits<int64_t>::max(), edges2explore, dag, mask, new_dag);
        if(fuse)
        {
            fill_kernels(new_dag, bhir.kernel_list);
            return;
        }
    }

    //Then we use the greedy algorithm to find a good initial guess
    {
        GraphDW new_dag(dag);
        fuse_greedy(new_dag);
        best_dag = new_dag.bglD();
        best_cost = dag_cost(best_dag);
    }

    if(mask.size() > 100)
    {
        cout << "FUSER-OPTIMAL: ABORT the size of the search space is too large: 2^";
        cout << mask.size() << "!" << endl;
        fill_kernels(best_dag, bhir.kernel_list);
        return;
    }
    else if(mask.size() > 10)
    {
        cout << "FUSER-OPTIMAL: the size of the search space is 2^" << mask.size() << "!" << endl;
    }

    fuse(dag, edges2explore, mask, 0, false);
    fuse(dag, edges2explore, mask, 0, true);
    fill_kernels(best_dag, bhir.kernel_list);
}

