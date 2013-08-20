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
#include <bh_vector.h>
#include <assert.h>


//***** Some private set functions *****/
inline static void nelem_set(void *vector, bh_intp val)
{
    bh_intp *v = ((bh_intp *)vector)-3;
    v[0] = val;
}
inline static void reserved_set(void *vector, bh_intp val)
{
    bh_intp *v = ((bh_intp *)vector)-3;
    v[1] = val;
}
inline static void elsize_set(void *vector, bh_intp val)
{
    bh_intp *v = ((bh_intp *)vector)-3;
    v[2] = val;
}

/* Returns the number of elements in the vector */
bh_intp bh_vector_nelem(const void *vector)
{
    bh_intp *v = ((bh_intp *)vector)-3;
    return v[0];
}

/* Returns the reverved number of elements */
bh_intp bh_vector_reserved(const void *vector)
{
    bh_intp *v = ((bh_intp *)vector)-3;
    return v[1];
}

/* Returns the size of each element in the vector (in bytes) */
bh_intp bh_vector_elsize(const void *vector)
{
    bh_intp *v = ((bh_intp *)vector)-3;
    return v[2];
}

/* Creates a new vector
 *
 * @elsize        The size of each element in the vector (in bytes)
 * @initial_size  The initial number of elements in the vector
 * @reserve_size  The number of reserved elements to allocate (incl. initial size)
 *                It must be greater or equal to the initial size
 * @return        The new vector as a void pointer, or NULL if out of memory
 */
void *bh_vector_create(bh_intp elsize, bh_intp initial_size,
                       bh_intp reserve_size)
{
    assert(elsize > 0);
    assert(initial_size >= 0);
    assert(initial_size <= reserve_size);

    bh_intp *v = (bh_intp *) malloc(3*sizeof(bh_intp) + elsize*reserve_size);
    if(v == NULL)
        return NULL;
    //We use the three first 64bit values to store:
    v[0] = initial_size;//Current number of elements
    v[1] = reserve_size;//Reserved number of elements (incl. initial size)
    v[2] = elsize;      //The size of each element in the vector (in bytes)
    return v+3;
}


/* De-allocate the vector
 *
 * @vector  The vector in question
 */
void bh_vector_destroy(void *vector)
{
   free(((bh_intp *)vector)-3);
}


/* Requests that the vector capacity be at least enough to contain n elements.
 *
 * If n is greater than the current vector capacity, the function causes the
 * container to reallocate its storage increasing its capacity to n (or greater).
 *
 * In all other cases, the function call does not cause a reallocation and the
 * vector capacity is not affected.
 *
 * @vector   The vector in question (in/out-put)
 * @n        The number of elements to reserve
 * @return   The updated vector (potential re-allocated), or NULL if out
 *           of memory
 */
void *bh_vector_reserve(void *vector, bh_intp n)
{
    if(bh_vector_reserved(vector) >= n)
        return vector;

    bh_intp *v = ((bh_intp *)vector)-3;
    v = (bh_intp *) realloc(v, 3*sizeof(bh_intp) + n*bh_vector_elsize(vector));
    if(v == NULL)
        return NULL;
    //Update vector and save the new reserve size
    vector = v+3;
    reserved_set(vector, n);
    return vector;
}


/* Resizes the container so that it contains n elements.
 *
 * If n is smaller than the current container size, the content is reduced
 * to its first n elements, removing those beyond (and destroying them).
 *
 * If n is greater than the current container size, the content is expanded
 * by inserting at the end as many elements as needed to reach a size of n.
 *
 * If n is also greater than the current container capacity, an automatic
 * reallocation of the allocated storage space takes place.
 *
 * Notice that this function changes the actual content of the container by
 * inserting or erasing elements from it.
 *
 * @vector   The vector in question
 * @size     The new size (in number of elements)
 * @return   The updated vector (potential re-allocated), or NULL if out
 *           of memory
 */
void *bh_vector_resize(void *vector, bh_intp size)
{
    vector = bh_vector_reserve(vector, size);
    if(vector == NULL)
        return NULL;

    nelem_set(vector, size);
    assert(bh_vector_nelem(vector) <= bh_vector_reserved(vector));
    return vector;
}


/* Adds a new element at the end of the vector, after its current last element.
 * The content of val is copied to the new element.
 *
 * @vector   The vector in question
 * @val      The element to extend with
 * @return   The updated vector (potential re-allocated), or NULL if out
 *           of memory
 */
void *bh_vector_push_back(void *vector, const void* val)
{
    bh_intp nelem  = bh_vector_nelem(vector);
    bh_intp elsize = bh_vector_elsize(vector);

    vector = bh_vector_resize(vector, nelem+1);
    if(vector == NULL)
        return NULL;
    //Copy the value to the last valid position in the vector
    memcpy(((char *)vector) + nelem * elsize, val, elsize);
    return vector;
}

