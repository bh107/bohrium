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

    //Initiate the process grid
    if((e = pgrid_init()) != CPHVB_SUCCESS)
        return e;

    if(pgrid_myrank != 0)
    {
        fprintf(stderr, "[VEM-CLUSTER] Multiple instants of the bridge is not allowed. "
        "Only rank-0 should execute the bridge. All slaves should execute cphvb_vem_cluster_slave.\n");
        MPI_Abort(MPI_COMM_WORLD,e);
    }

    //Send the component name
    if((e = dispatch_reset()) != CPHVB_SUCCESS)
        return e;
    if((e = dispatch_add2payload(strlen(self->name)+1, self->name)) != CPHVB_SUCCESS)
        return e;
    if((e = dispatch_send(CPHVB_CLUSTER_DISPATCH_INIT)) != CPHVB_SUCCESS)
        return e;

    //Execute our self
    if((e = exec_init(self->name)) != CPHVB_SUCCESS)
        return e;

    return CPHVB_SUCCESS; 
}


cphvb_error cphvb_vem_cluster_shutdown(void)
{
    cphvb_error e;

    //Send the component name
    if((e = dispatch_reset()) != CPHVB_SUCCESS)
        return e;
    if((e = dispatch_send(CPHVB_CLUSTER_DISPATCH_SHUTDOWN)) != CPHVB_SUCCESS)
        return e;
    
    //Execute our self
    return exec_shutdown();
} 

    
cphvb_error cphvb_vem_cluster_reg_func(char *fun, cphvb_intp *id)
{
    cphvb_error e;
    assert(fun != NULL);

    //The Master will find the new 'id' if required.
    if((e = exec_reg_func(fun, id)) != CPHVB_SUCCESS)
        return e;

    //Send the component name
    if((e = dispatch_reset()) != CPHVB_SUCCESS)
        return e;
    if((e = dispatch_add2payload(sizeof(cphvb_intp), id)) != CPHVB_SUCCESS)
        return e;
    if((e = dispatch_add2payload(strlen(fun)+1, fun)) != CPHVB_SUCCESS)
        return e;
    if((e = dispatch_send(CPHVB_CLUSTER_DISPATCH_UFUNC)) != CPHVB_SUCCESS)
        return e;

    return CPHVB_SUCCESS;
}


cphvb_error cphvb_vem_cluster_execute(cphvb_intp count,
                                      cphvb_instruction inst_list[])
{
    cphvb_error e;
//    cphvb_pprint_instr_list(inst_list, count, "MASTER");

    //Send the instruction list and operands to the slaves
    if((e = dispatch_inst_list(count, inst_list)) != CPHVB_SUCCESS)
        return e;
    
    
    return exec_execute(count,inst_list);
}
