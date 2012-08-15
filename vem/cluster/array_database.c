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

#include "array_database.h"
#include "cphvb_vem_cluster.h"

//Array-bases belonging to local MPI process
static dndarray *dndarrays[DNPY_MAX_NARRAYS];

/*===================================================================
 *
 * Initiate the local array database.
 */
void arydb_init(void)
{
    memset(dndarrays, 0, sizeof(dndarray*));

} /* arydb_init */

/*===================================================================
 *
 * Put, get & remove array from the local array database.
 */
void arydb_put(dndarray *ary)
{
    cphvb_intp i;

    for(i=0; i < DNPY_MAX_NARRAYS; i++)
        if(dndarrays[i] == NULL)
        {
            dndarrays[i] = ary;
            return;
        }
    fprintf(stderr, "arydb_put, MAX_NARRAYS is exceeded\n");
    MPI_Abort(MPI_COMM_WORLD, -1);
    return;
}
dndarray *arydb_get(cphvb_array *ary)
{
    cphvb_intp i;
    if(ary != NULL)
        for(i=0; i < DNPY_MAX_NARRAYS; i++)
            if(dndarrays[i] != NULL &&
               dndarrays[i]->cphvb_ary == ary)
                return dndarrays[i];
    fprintf(stderr, "get_dndarray %p does not exist\n", ary);
    MPI_Abort(MPI_COMM_WORLD, -1);
    return NULL;
}
void arydb_rm(cphvb_array *ary)
{
    cphvb_intp i;
    if(ary != NULL)
        for(i=0; i < DNPY_MAX_NARRAYS; i++)
            if(dndarrays[i] != NULL &&
               dndarrays[i]->cphvb_ary == ary)
            {
                dndarray *a = dndarrays[i];
                //Cleanup base.
                dndarrays[i] = NULL;

                MPI_Type_free(&a->mpi_dtype);
                free(a->rootnodes);
                return;
            }
    fprintf(stderr, "rm_dndarray %p does not exist\n", ary);
    MPI_Abort(MPI_COMM_WORLD, -1);
    return;
}


