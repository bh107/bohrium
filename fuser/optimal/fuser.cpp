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

using namespace std;
using namespace boost;
using namespace bohrium::dag;

bool fuse_mask(const Graph &dag, const vector<EdgeW> &edges2explore,
               const vector<bool> &mask, Graph &new_dag)
{
    vector<EdgeW> edges2merge;
    unsigned int i=0;
    BOOST_FOREACH(const EdgeW &e, edges2explore)
    {
        if(mask[i++])
        {
            edges2merge.push_back(e);
        }
    }
    try
    {
        merge_vertices(dag, edges2merge, new_dag, true);
    }
    catch (const runtime_error &e)
    {
        return false;
    }

    if(cycles(new_dag))
        return false;

    return true;
}

int fuser_count=0;
uint64_t best_cost;
Graph best_dag;
void fuse(const Graph &dag, const vector<EdgeW> &edges2explore,
          vector<bool> mask, unsigned int offset, bool merge_next)
{
    if(not merge_next)
    {
        Graph new_dag;
        mask[offset] = merge_next;
        const bool fusible = fuse_mask(dag, edges2explore, mask, new_dag);
        const uint64_t cost = dag_cost(new_dag);
        if(cost >= best_cost)
            return;
        if(fusible)
        {
            best_cost = cost;
            best_dag = new_dag;
            #ifdef VERBOSE
                std::stringstream ss;
                ss << "new_best_dag-" << fuser_count << "-" << dag_cost(new_dag) << ".dot";
                printf("write file: %s\n", ss.str().c_str());
                pprint(new_dag, ss.str().c_str());
            #endif
            return;
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
    from_bhir(bhir, dag);
    transitive_reduction(dag);
    fuse_gentle(dag);

    //The list of edges that we should try to merge
    vector<EdgeW> edges2explore;
    all_weights(dag, edges2explore);

    if(edges2explore.size() == 0)
    {
        fill_kernels(dag, bhir.kernel_list);
        return;
    }

    //First we check the trivial case where all kernels are merged
    vector<bool> mask(edges2explore.size(), true);
    {
        Graph new_dag;
        if(fuse_mask(dag, edges2explore, mask, new_dag))
        {
            fill_kernels(new_dag, bhir.kernel_list);
            return;
        }
    }

    //Then we use the greedy algorithm to find a good initial guess
    best_dag = dag;
    fuse_greedy(best_dag);
    best_cost = dag_cost(best_dag);

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

    fuse(dag, edges2explore, mask, 0, true);
    fuse(dag, edges2explore, mask, 0, false);
    fill_kernels(best_dag, bhir.kernel_list);
}

