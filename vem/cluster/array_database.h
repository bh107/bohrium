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

/*
 * There is a local array database on each MPI-process.
 * The database consist of all distributed array-bases.
 */

#include "cphvb_vem_cluster.h"

#ifndef ARRAY_DATABASE_H
#define ARRAY_DATABASE_H
#ifdef __cplusplus
extern "C" {
#endif

/*===================================================================
 *
 * Initiate the local array database.
 */
void arydb_init(void);


/*===================================================================
 *
 * Put, get & remove arrays from the local array database.
 */
void arydb_put(dndarray *ary);
dndarray *arydb_get(cphvb_array *ary);
void arydb_rm(cphvb_array *ary);


#ifdef __cplusplus
}
#endif

#endif /* !defined(ARRAY_DATABASE_H) */
