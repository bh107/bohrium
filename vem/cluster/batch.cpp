/*
This file is part of cphVB and copyright (c) 2012 the cphVB team:
http://cphvb.bitbucket.org

cphVB is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as 
published by the Free Software Foundation, either version 3 
of the License, or (at your option) any later version.

cphVB is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the 
GNU Lesser General Public License along with cphVB. 

If not, see <http://www.gnu.org/licenses/>.
*/

#include <cassert>
#include <StaticStore.hpp>
#include <cphvb.h>
#include <vector>
#include "task.h"
#include "exec.h"
#include "except.h"
#include "pgrid.h"

static StaticStore<cphvb_array> tmp_ary_store(512);
static std::vector<task> task_store;

/* Returns a temporary array that will be freed on a batch flush
 * 
 * @return The new temporary array
 */
cphvb_array* batch_tmp_ary()
{
    return tmp_ary_store.c_next();
}


/* Schedule an task
 * @t      The task to schedule 
 * @return The new temporary array
 */
void batch_schedule(const task& t)
{
    task_store.push_back(t);
}


/* Flush all scheduled instructions
 * 
 */
void batch_flush()
{
    for(std::vector<task>::iterator it=task_store.begin(); 
        it != task_store.end(); ++it)
    {
        switch((*it).inst.type) 
        {
            case TASK_INST:
            {
                cphvb_error e;
                cphvb_instruction *inst = &((*it).inst.inst);
                if((e = exec_vem_execute(1, inst)) != CPHVB_SUCCESS)
                    EXCEPT_INST(inst->opcode, e, inst->status);
                break;
            }
            default:
            {
                fprintf(stderr, "[VEM-CLUSTER] batch_schedule encountered "
                        "an unknown task type\n");
                MPI_Abort(MPI_COMM_WORLD,CPHVB_ERROR);
            }
        }        
    }
    task_store.clear();
    tmp_ary_store.clear(); 
}




