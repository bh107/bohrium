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

#ifndef __CPHVB_VEM_CLUSTER_TMP_H
#define __CPHVB_VEM_CLUSTER_TMP_H


/* Returns a temporary array that will be de-allocated  
 * on tmp_clear().
 * 
 * @return The temporary array
 */
cphvb_array* tmp_get_ary();


/* Returns temporary memory for miscellaneous use
 * that will be de-allocated on tmp_clear().
 * 
 * @return The temporary memory
 */
void* tmp_get_misc(cphvb_intp size);


/* Clear all temporary data structures
 */
void tmp_clear();

#endif
