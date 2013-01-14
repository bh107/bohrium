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
#include "task.h"

#ifndef __CPHVB_VEM_CLUSTER_BATCH_H
#define __CPHVB_VEM_CLUSTER_BATCH_H

/* Returns a temporary array that will be freed on a batch flush
 * 
 * @return The new temporary array
 */
cphvb_array* batch_tmp_ary();


/* Schedule an task
 * @t      The task to schedule 
 * @return The new temporary array
 */
void batch_schedule(const task& t);


/* Flush all scheduled instructions
 * 
 */
void batch_flush();


#endif
