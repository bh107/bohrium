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

void fuser(bh_ir &bhir)
{
    Graph dag;
    bohrium::dag::from_bhir(bhir, dag);
    bohrium::dag::transitive_reduction(dag);
    bohrium::dag::fuse_gentle(dag);
    bohrium::dag::fuse_greedy(dag);
    bohrium::dag::fill_kernels(dag, bhir.kernel_list);
}
