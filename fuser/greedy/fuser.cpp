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
#include <boost/foreach.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/topological_sort.hpp>
#include <vector>
#include <set>
#include <iterator>

using namespace std;
using namespace boost;

typedef adjacency_list<setS, vecS, bidirectionalS, bh_ir_kernel> Graph;
typedef graph_traits<Graph>::vertex_descriptor Vertex;
typedef graph_traits<Graph>::edge_descriptor Edge;

void fuser(bh_ir &bhir)
{
    Graph dag;
    bh_dag_from_bhir(bhir, dag);
    bh_dag_transitive_reduction(dag);
    bh_dag_fuse_gentle(dag);

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

    //Lets fill the bhir;
    vector<Vertex> topological_order;
    topological_sort(dag, back_inserter(topological_order));
    BOOST_REVERSE_FOREACH(const Vertex &v, topological_order)
    {
        bhir.kernel_list.push_back(dag[v]);
    }
}
