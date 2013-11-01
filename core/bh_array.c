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

/** Create a new base array.
 *
 * @param type The type of data in the array
 * @param nelements The number of elements
 * @param new_base The handler for the newly created base
 * @return Error code (BH_SUCCESS, BH_OUT_OF_MEMORY)
 */
bh_error bh_create_base(bh_type    type,
                        bh_index   nelements,
                        bh_base**  new_base)
{

    bh_base *base = (bh_base *) malloc(sizeof(bh_base));
    if(base == NULL)
        return BH_OUT_OF_MEMORY;
    base->type = type;
    base->nelem = nelements;
    base->data = NULL;
    *new_base = base;

    return BH_SUCCESS;
}


/** Destroy the base array.
 *
 * @param base  The base array in question
 */
void bh_destroy_base(bh_base**  base)
{
    bh_base *b = *base;
    free(b);
    b = NULL;
}
