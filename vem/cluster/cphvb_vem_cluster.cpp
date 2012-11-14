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
#include <cstring>
#include <iostream>
#include <vector>
#include <cphvb.h>

#include "cphvb_vem_cluster.h"
#include "exec.h"
#include "dispatch.h"
#include "pgrid.h"


cphvb_error cphvb_vem_cluster_init(cphvb_component *self)
{
    cphvb_error e;
    cphvb_intp size;
    dispatch_msg *msg;

    //Initiate the process grid
    if((e = pgrid_init()) != CPHVB_SUCCESS)
        return e;

    //Initiate the dispatch system
    if((e = dispatch_init()) != CPHVB_SUCCESS)
        return e;

    if(pgrid_myrank != 0)
    {
        fprintf(stderr, "[VEM-CLUSTER] Multiple instants of the bridge is not allowed. "
        "Only rank-0 should execute the bridge. All slaves should execute cphvb_vem_cluster_slave.\n");
        MPI_Abort(MPI_COMM_WORLD,e);
    }

    //Create dispatch message
    msg = (dispatch_msg*) malloc(sizeof(dispatch_msg) + sizeof(char)*CPHVB_COMPONENT_NAME_SIZE);
    if(msg == NULL)
        return CPHVB_OUT_OF_MEMORY;

    //We we send the component name
    size = strnlen(self->name, CPHVB_COMPONENT_NAME_SIZE) * sizeof(char);
    e = dispatch_send(CPHVB_CLUSTER_DISPATCH_INIT, size, self->name);
    free(msg);
    
    if(e != CPHVB_SUCCESS)
        return e;

    if((e = exec_init(self->name)) != CPHVB_SUCCESS)
        return e;

    return CPHVB_SUCCESS; 
}


cphvb_error cphvb_vem_cluster_shutdown(void)
{
    cphvb_error e;
    if((e = dispatch_send(CPHVB_CLUSTER_DISPATCH_SHUTDOWN, 0, NULL)) != CPHVB_SUCCESS)
        return e;
    return exec_shutdown();
} 

    
cphvb_error cphvb_vem_cluster_reg_func(char *fun, cphvb_intp *id)
{
    //TODO: Send to slaves

    return exec_reg_func(fun, id);
}


cphvb_error cphvb_vem_cluster_execute(cphvb_intp count,
                                      cphvb_instruction inst_list[])
{
    //TODO: Send to slaves

    return exec_execute(count,inst_list);
}
