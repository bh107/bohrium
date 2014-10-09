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
#include <map>
#include <iterator>

using namespace std;
using namespace boost;

typedef adjacency_list<setS, vecS, bidirectionalS, bh_ir_kernel> Graph;
typedef graph_traits<Graph>::vertex_descriptor Vertex;
typedef graph_traits<Graph>::edge_descriptor Edge;

void fuser(bh_ir &bhir)
{
    if(bhir.kernel_list.size() != 0)
    {
        throw logic_error("The kernel_list is not empty!");
    }

    //First we create singleton kernels
    Graph dag;
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
    bh_dag_transitive_reduction(dag);
    bh_dag_fuse_gentle(dag);

    //Lets fill the bhir;
    vector<Vertex> topological_order;
    topological_sort(dag, back_inserter(topological_order));
    BOOST_REVERSE_FOREACH(const Vertex &v, topological_order)
    {
        bhir.kernel_list.push_back(dag[v]);
    }
}

