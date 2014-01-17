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
#include <sstream>
#include <iostream>
#include <string>
#include <vector>
#include <set>
#include <bh.h>

using namespace std;

typedef enum { NONE, RANGE, FUSABLE, STREAMABLE } tag_t;

/*
bh_error void ref_count(bh_ir* bhir, dag)
{
    for(bh_intp i=0; i< bhir->dag_list[dag].nnode; ++i) {
        bh_intp id = bhir->dag_list[dag].node_map[i];
        if (id < 0) {
            id = -1*id-1;
            bh_error err = ref_count(bhir, );
            if (err != BH_SUCCESS)
                return err;
        } else {
            bh_error err = func(&bhir->instr_list[id]);
            if (err != BH_SUCCESS)
                return err;
        }
    }
    return BH_SUCCESS;
}*/

void range_filter(bh_ir* bhir)
{
    cout << "### range-filter on bhir->ninstr = " << bhir->ninstr << endl;
    for(bh_intp dag=0; dag<bhir->ndag; ++dag) {

        cout << "Looking at dag[" << dag << "] " << dag+1 << " of "<< bhir->ndag << \
                ", containing " << bhir->dag_list[dag].nnode << " node(s)" << endl;

        if (bhir->dag_list[dag].tag != NONE) {      // Already tagged, skipping it.
            cout << "Skipping it..." << endl;
            continue;
        }

        for(bh_intp max_nodes=bhir->dag_list[dag].nnode; max_nodes>0; max_nodes--) {

            bh_intp nodes[] = {0};
            bh_dag_split(bhir, 1, nodes, dag, -1);  // Sub-graphing
            bh_intp dag_idx = bhir->ndag-1;         // Index of sub-graph
            bhir->dag_list[dag_idx].tag = RANGE;    // Tagging

        }
    }
    cout << "###" << endl;
}

