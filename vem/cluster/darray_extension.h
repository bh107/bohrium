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

#ifndef __CPHVB_VEM_CLUSTER_ARRAY_EXTENSION_H
#define __CPHVB_VEM_CLUSTER_ARRAY_EXTENSION_H

#ifdef __cplusplus
extern "C" {
#endif


//Extension to the cphvb_array for cluster information
typedef struct
{
    //Process rank that owns the array.
    int rank;
}darray_ext;


//Extension to the cphvb_array for cluster information
typedef struct
{
    //The id of the array. This is identical with the array-struct address 
    //on the master-process.
    cphvb_intp id;
    //The global array-struct.
    cphvb_array global_ary;
}darray;


#ifdef __cplusplus
}
#endif

#endif
