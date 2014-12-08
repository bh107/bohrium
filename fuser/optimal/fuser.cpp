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

bool fuse_mask(const vector<EdgeW> &edges2explore,
               const vector<bool> &mask, GraphDW &dag)
{
    /*
    vector<EdgeW> edges2merge;
    unsigned int i=0;
    BOOST_FOREACH(const EdgeW &e, edges2explore)
    {
        if(mask[i++])
        {
            edges2merge.push_back(e);
        }
    }
    if(not merge_vertices(dag, edges2merge))
        return false;
    if(cycles(dag.bglD()))
        return false;
    */
    return true;
}
#ifdef VERBOSE
double  purge_count=0;
uint64_t explore_count=0;
int fuser_count=0;
#endif
uint64_t best_cost;
GraphDW best_dag;
void fuse(const GraphDW &dag, const vector<EdgeW> &edges2explore,
          vector<bool> mask, unsigned int offset, bool merge_next)
{
    if(not merge_next)
    {
#ifdef VERBOSE
        ++explore_count;
        if(explore_count%1000 == 0)
        {
            cout << "purge count: " << purge_count << " / " << pow(2.0,mask.size()) << endl;
            cout << "explore count: " << explore_count << endl;
        }
#endif
        GraphDW new_dag(dag);
        mask[offset] = merge_next;
        const bool fusible = fuse_mask(edges2explore, mask, new_dag);
        const uint64_t cost = dag_cost(new_dag.bglD());
        if(cost >= best_cost)
        {
#ifdef VERBOSE
            purge_count += pow(2.0, mask.size()-offset-1);
#endif
            return;
        }
        if(fusible)
        {
            best_cost = cost;
            best_dag = new_dag;
#ifdef VERBOSE
            std::stringstream ss;
            ss << "new_best_dag-" << fuser_count << "-" << dag_cost(new_dag.bglD()) << ".dot";
            printf("write file: %s\n", ss.str().c_str());
            pprint(new_dag, ss.str().c_str());
            purge_count += pow(2.0, mask.size()-offset-1);
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
    reverse(edges2explore.begin(), edges2explore.end());

    if(edges2explore.size() == 0)
    {
        fill_kernels(dag.bglD(), bhir.kernel_list);
        return;
    }

    //First we check the trivial case where all kernels are merged
    vector<bool> mask(edges2explore.size(), true);
    {
        GraphDW new_dag(dag);
        if(fuse_mask(edges2explore, mask, new_dag))
        {
            fill_kernels(new_dag.bglD(), bhir.kernel_list);
            return;
        }
    }

    //Then we use the greedy algorithm to find a good initial guess
    best_dag = dag;
    fuse_greedy(best_dag);
    best_cost = dag_cost(best_dag.bglD());

    if(mask.size() > 100)
    {
        cout << "FUSER-OPTIMAL: ABORT the size of the search space is too large: 2^";
        cout << mask.size() << "!" << endl;
        fill_kernels(best_dag.bglD(), bhir.kernel_list);
        return;
    }
    else if(mask.size() > 10)
    {
        cout << "FUSER-OPTIMAL: the size of the search space is 2^" << mask.size() << "!" << endl;
    }

    fuse(dag, edges2explore, mask, 0, true);
    fuse(dag, edges2explore, mask, 0, false);
    fill_kernels(best_dag.bglD(), bhir.kernel_list);
}

