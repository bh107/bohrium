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
#ifndef __BH_DYNAMIC_LIST_H
#define __BH_DYNAMIC_LIST_H

#include <bh.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    // The size of a single element
    bh_intp elsize;
    // The number of elements used
    bh_intp count;
    // The number of elements allocated
    bh_intp allocated;
    // The actual data storage
    void* data;
} bh_dynamic_list;

/* Creates a new dynamic list
 * @elsize The size of each element in the list
 * @initial_size The initial number of elements to allocate
 * @return The new dynamic list, or NULL if the list could not be created
 */
DLLEXPORT bh_dynamic_list* bh_dynamic_list_create(bh_intp elsize, bh_intp initial_size);

/* Appends an element to the list
 * @list The list to append to
 * @return A pointer to the storage, or -1 if out of memory or the list is invalid
 */
DLLEXPORT bh_intp bh_dynamic_list_append(bh_dynamic_list* list);

/* Removes an element from the list
 * @list The list to remove from
 */
DLLEXPORT void bh_dynamic_list_remove(bh_dynamic_list* list, bh_intp index);

/* Frees all resources allocated by the list
 * @list The list to destroy
 */
DLLEXPORT void bh_dynamic_list_destroy(bh_dynamic_list* list);

#ifdef __cplusplus
}
#endif

#endif

