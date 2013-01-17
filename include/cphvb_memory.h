/*
This file is part of Bohrium and copyright (c) 2012 the Bohrium
team <http://www.bh107.org>.

Bohrium is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as 
published by the Free Software Foundation, either version 3 
of the License, or (at your option) any later version.

Bohrium is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the 
GNU Lesser General Public License along with Bohrium. 

If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __CPHVB_MEMORY_H
#define __CPHVB_MEMORY_H

#ifdef __cplusplus
extern "C" {
#endif

#include <cphvb_type.h>

/* Allocate an alligned contigous block of memory,
 * without any initialization
 *
 * @size  The size of the allocated block
 * @return A pointer to data, and NULL on error
 */
cphvb_data_ptr cphvb_memory_malloc(cphvb_intp size);

/* Frees a previously allocated data block
 *
 * @data  The pointer returned from a call to cphvb_memory_malloc
 * @size  The size of the allocated block
 * @return A pointer to data, and NULL on error
 */
cphvb_intp cphvb_memory_free(cphvb_data_ptr data, cphvb_intp size);

#ifdef __cplusplus
}
#endif

#endif