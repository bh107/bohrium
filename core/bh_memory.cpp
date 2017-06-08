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

#ifdef _WIN32
#include <malloc.h>
#else
#include <sys/mman.h>
#endif
#include <cstddef>

#include <bh_memory.h>
#include <bh_win.h>

/* Allocate an alligned contigous block of memory,
 * does not apply any initialization
 *
 * @size  The size of the allocated block
 * @return A pointer to data, and NULL on error
 */
void* bh_memory_malloc(int64_t size)
{
#ifdef _WIN32
    return _aligned_malloc(size, 16);
#else
    //Allocate page-size aligned memory.
    //The MAP_PRIVATE and MAP_ANONYMOUS flags is not 100% portable. See:
    //<http://stackoverflow.com/questions/4779188/how-to-use-mmap-to-allocate-a-memory-in-heap>
    void* data = mmap(0, size, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    if(data == MAP_FAILED)
        return NULL;
    else
        return data;
#endif
}

/* Frees a previously allocated data block
 *
 * @data  The pointer returned from a call to bh_memory_malloc
 * @size  The size of the allocated block
 * @return A pointer to data, and NULL on error
 */
int64_t bh_memory_free(void* data, int64_t size)
{
#ifdef _WIN32
	_aligned_free(data);
	return 0;
#else
	return munmap(data, size);
#endif
}
