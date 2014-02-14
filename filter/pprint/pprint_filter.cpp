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
#include <bh_flow.h>
#include <stdio.h>

using namespace std;

static int pprint_filter_count=0;
void pprint_filter(bh_ir *bhir)
{
    char graph_fn[8000];
    char trace_fn[8000];
    char flowd_fn[8000];
    char flowh_fn[8000];

    ++pprint_filter_count;
    snprintf(graph_fn, 8000, "graph-%d.dot", pprint_filter_count);
    snprintf(trace_fn, 8000, "trace-%d.txt", pprint_filter_count);
    snprintf(flowd_fn, 8000, "flow-%d.dot",  pprint_filter_count);
    snprintf(flowh_fn, 8000, "flow-%d.html",  pprint_filter_count);

    printf(
        "pprint-filter: writing graph('%s'), trace('%s'), and flow('%s', '%s').\n",
        graph_fn, trace_fn, flowd_fn, flowh_fn
    );

    bh_bhir2dot(bhir, graph_fn);            // Graph
    bh_pprint_trace_file(bhir, trace_fn);   // Trace

    printf("Warning - the flow is generated based on the original instruction list\n");
    bh_flow flow = bh_flow(bhir->ninstr, bhir->instr_list);
    flow.dot(flowd_fn);
    flow.html(flowh_fn);
}

