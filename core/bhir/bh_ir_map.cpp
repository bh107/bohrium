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

/* This is a collection of useful map functions for the BhIR */

#include <bh.h>
#include <assert.h>


/* Applies the 'func' on each instruction in the 'dag' and it's
 * sub-dags topologically.
 *
 * @bhir        The BhIR handle
 * @dag         The dag to start the map from
 * @func        The func to call with each instructions
 * @return      Error code (BH_SUCCESS, BH_ERROR)
 */
bh_error bh_ir_map_instr(bh_ir *bhir, bh_dag *dag,
                         bh_ir_map_instr_func func)
{
    for(bh_intp i=0; i<dag->nnode; ++i)
    {
        bh_intp id = dag->node_map[i];
        if(id < 0)
        {
            id = -1*id-1;
            bh_error err = bh_ir_map_instr(bhir, &bhir->dag_list[id], func);
            if(err != BH_SUCCESS)
                return err;
        }
        else
        {
            bh_error err = func(&bhir->instr_list[id]);
            if(err != BH_SUCCESS)
                return err;
        }
    }
    return BH_SUCCESS;
}
