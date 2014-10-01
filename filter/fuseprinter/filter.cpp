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

static int filter_count=0;
void filter(const bh_ir bhir)
{
    char dag_fn[8000];

    snprintf(dag_fn, 8000, "dag-%d.dot", ++filter_count);
    printf("fuseprinter: writing dag('%s').\n", dag_fn);

    bhir.pprint_kernel_dag(dag_fn);   // Trace
}

