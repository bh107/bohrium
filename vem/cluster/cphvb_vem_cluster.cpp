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


cphvb_error cphvb_vem_cluster_init(cphvb_component *self)
{
    //TODO: Send to slaves

    return exec_init(self);
}


cphvb_error cphvb_vem_cluster_shutdown(void)
{
    //TODO: Send to slaves

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
