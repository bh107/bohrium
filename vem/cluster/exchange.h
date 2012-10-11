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

#ifndef __CPHVB_VEM_CLUSTER_EXCHANGE_H

#include <cphvb.h>

void exchange_inst_bridge2vem(cphvb_intp count,
                              const cphvb_instruction bridge_inst[],
                              cphvb_instruction vem_inst[]);

void exchange_inst_vem2bridge(cphvb_intp count,
                              const cphvb_instruction vem_inst[],
                              cphvb_instruction bridge_inst[]);

void exchange_inst_discard(cphvb_array *vem_ary);

#endif
