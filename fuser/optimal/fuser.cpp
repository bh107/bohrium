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
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/topological_sort.hpp>
#include <vector>
#include <map>
#include <iterator>

using namespace std;
using namespace boost;

typedef adjacency_list<setS, vecS, bidirectionalS, bh_ir_kernel> Graph;
typedef graph_traits<Graph>::vertex_descriptor Vertex;
typedef graph_traits<Graph>::edge_descriptor Edge;

void dag2kernel_list(const Graph &dag, vector<bh_ir_kernel> &kernel_list)
{
    vector<Vertex> topological_order;
    topological_sort(dag, back_inserter(topological_order));
    BOOST_REVERSE_FOREACH(const Vertex &v, topological_order)
    {
        kernel_list.push_back(dag[v]);
    }
}

bool fuse_mask(const Graph &dag, const vector<Edge> &edges2explore,
               const vector<bool> &mask, Graph &new_dag)
{
    vector<Edge> edges2merge;
    unsigned int i=0;
    BOOST_FOREACH(const Edge &e, edges2explore)
    {
        if(mask[i++])
        {
            edges2merge.push_back(e);
        }
    }
    try
    {
        bh_dag_merge_vertices(dag, edges2merge, new_dag, true);
    }
    catch (const runtime_error &e)
    {
        return false;
    }

    if(bh_dag_cycles(new_dag))
        return false;

    return true;
}

int fuser_count=0;
uint64_t best_cost = numeric_limits<uint64_t>().max();
Graph best_dag;
void fuse(const Graph &dag, const vector<Edge> &edges2explore,
          vector<bool> mask, unsigned int offset, bool merge_next)
{
    if(not merge_next)
    {
        Graph new_dag;
        mask[offset] = merge_next;
        if(fuse_mask(dag, edges2explore, mask, new_dag))
        {
            const uint64_t c = bh_dag_cost(new_dag);
            if(best_cost > c)
            {
                best_cost = c;
                best_dag = new_dag;
#ifdef VERBOSE
                std::stringstream ss;
                ss << "new_best_dag-" << fuser_count << "-" << bh_dag_cost(new_dag) << ".dot";
                printf("write file: %s\n", ss.str().c_str());
                bh_dag_pprint(new_dag, ss.str().c_str());
#endif
            }
        }
    }
    if(offset+1 < mask.size())
    {
        fuse(dag, edges2explore, mask, offset+1, true);
        fuse(dag, edges2explore, mask, offset+1, false);
    }
}

void fuser(bh_ir &bhir)
{
    Graph dag;
    bh_dag_from_bhir(bhir, dag);
    bh_dag_transitive_reduction(dag);
    bh_dag_fuse_gentle(dag);

    //The list of edges that we should try to merge
    vector<Edge> edges2explore;
    BOOST_FOREACH(const Edge &e, edges(dag))
    {
        if(dag[target(e,dag)].fusible(dag[source(e,dag)]))
            edges2explore.push_back(e);
    }

    vector<bool> mask(edges2explore.size(), true);
    if(mask.size() == 0)
        return;

    //First we check the trivial case where all kernels are merged
    {
        Graph new_dag;
        if(fuse_mask(dag, edges2explore, mask, new_dag))
        {
            dag2kernel_list(new_dag, bhir.kernel_list);
            return;
        }
    }

    if(mask.size() > 20)
    {
        cout << "FUSER-OPTIMAL: ABORT the size of the search space is too large: 2^";
        cout << mask.size() << "!" << endl;
        return;
    }
    else if(mask.size() > 10)
    {
        cout << "FUSER-OPTIMAL: the size of the search space is 2^" << mask.size() << "!" << endl;
    }

    fuse(dag, edges2explore, mask, 0, true);
    fuse(dag, edges2explore, mask, 0, false);

    if(best_cost < numeric_limits<uint64_t>().max())
    {
        dag2kernel_list(best_dag, bhir.kernel_list);
    }
}

