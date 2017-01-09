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

#include <cassert>
#include <deque>
#include <StaticStore.hpp>
#include "tmp.h"
#include "except.h"


//Temporary array store
static StaticStore<bh_base> ary_store(512);

//Temporary miscellaneous stores
static std::deque<char *> misc_stores;


/* Returns a temporary array that will be de-allocated
 * on tmp_clear().
 *
 * @type   The new array data type
 * @nelem  Number of elements
 * @return The temporary array
 */
bh_base* tmp_get_ary(bh_type type, bh_index nelem)
{
    bh_base *ary = ary_store.c_next();
    ary->type = type;
    ary->nelem = nelem;
    ary->data = NULL;
    return ary;
}


/* Returns temporary memory for miscellaneous use
 * that will be de-allocated on tmp_clear().
 *
 * @return The temporary memory
 */
void* tmp_get_misc(bh_intp size)
{
    char *t = (char*) malloc(size);
    if(t == NULL)
        EXCEPT_OUT_OF_MEMORY();
    misc_stores.push_back(t);
    return t;
}


/* Clear all temporary data structures
 */
void tmp_clear()
{
    ary_store.clear();

    //Free all misc stores
    std::deque<char*>::iterator it = misc_stores.begin();

    while(it != misc_stores.end())
        free(*it++);
    misc_stores.clear();
}


