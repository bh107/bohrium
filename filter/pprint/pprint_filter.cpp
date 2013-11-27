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
#include <stdio.h>

using namespace std;

static int count=0;
void pprint_filter(bh_ir *bhir)
{
    char graph_fn[8000];
    char trace_fn[8000];

    ++count;
    snprintf(graph_fn, 8000, "graph-%d.dot", count);
    snprintf(trace_fn, 8000, "trace-%d.txt", count);

    printf(
        "pprint-filter: writing graph('%s') and trace('%s').\n",
        graph_fn, trace_fn
    );

    bh_bhir2dot(bhir, graph_fn);            // Graph
    bh_pprint_trace_file(bhir, trace_fn);   // Trace
}

