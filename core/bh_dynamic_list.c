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

#include <bh.h>
#include <bh_dynamic_list.h>

/* Creates a new dynamic list
 * @elsize The size of each element in the list
 * @initial_size The initial number of elements to allocate
 * @return The new dynamic list, or NULL if the list could not be created
 */
bh_dynamic_list* bh_dynamic_list_create(bh_intp elsize, bh_intp initial_size)
{
    if (elsize == 0)
        return NULL;
        
    bh_dynamic_list* list = (bh_dynamic_list*)malloc(sizeof(bh_dynamic_list));
    if (list == NULL)
        return NULL;
    
    list->elsize = elsize;
    list->count = 0;
    list->allocated = initial_size;
    list->data = malloc(elsize * initial_size);
    if (list->data == NULL)
    {
        free(list);
        return NULL;
    }
    
    return list;
}

/* Appends an element to the list
 * @list The list to append to
 * @return A pointer to the storage, or -1 if out of memory or the list is invalid
 */
bh_intp bh_dynamic_list_append(bh_dynamic_list* list)
{
    if (list->elsize <= 0)
        return -1;
        
    if (list->count >= list->allocated)
    {
        bh_intp newsize = list->allocated * 2;
        void* newdata = realloc(list->data, newsize * list->elsize);
        if (newdata == NULL)
            return -1;
            
        list->data = newdata;
        list->allocated = newsize;
    }
    
    return list->count++;
}

/* Removes an element from the list
 * @list The list to remove from
 */
DLLEXPORT void bh_dynamic_list_remove(bh_dynamic_list* list, bh_intp index)
{
    //Simplistic, we only remove the tail,
    // and accept some memory inefficiencies otherwise
    
    if (index == list->count - 1)
        list->count--;
        
    // We can add a "list of removed elements" as well,
    // if this gets too wasteful
}

/* Frees all resources allocated by the list
 * @list The list to destroy
 */
void bh_dynamic_list_destroy(bh_dynamic_list* list)
{
    free(list->data);
    list->data = NULL;
    list->count = -1;
    list->allocated = -1;
    list->elsize = 0;
    free(list);
}