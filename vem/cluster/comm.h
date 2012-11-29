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

#include <cphvb.h>

#ifndef __CPHVB_VEM_CLUSTER_COMM_H
#define __CPHVB_VEM_CLUSTER_COMM_H


/* Distribute the global array data to all slave processes.
 * NB: this is a collective operation.
 * 
 * @global_ary Global base array
 */
cphvb_error comm_master2slaves(cphvb_array *global_ary);


/* Gather the global array data to all slave processes.
 * NB: this is a collective operation.
 * 
 * @global_ary Global base array
 */
cphvb_error comm_slaves2master(cphvb_array *global_ary);


#endif
