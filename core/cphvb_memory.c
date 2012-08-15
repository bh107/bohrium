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

#ifdef _WIN32
#include <malloc.h>
#else
#include <sys/mman.h>
#endif

/* Allocate an alligned contigous block of memory,
 * does not apply any initialization
 *
 * @size  The size of the allocated block
 * @return A pointer to data, and NULL on error
 */
cphvb_data_ptr cphvb_memory_malloc(cphvb_intp size)
{
#ifdef _WIN32
	return _aligned_malloc(size, 16);
#else
    //Allocate page-size aligned memory.
    //The MAP_PRIVATE and MAP_ANONYMOUS flags is not 100% portable. See:
    //<http://stackoverflow.com/questions/4779188/how-to-use-mmap-to-allocate-a-memory-in-heap>

	cphvb_data_ptr data = mmap(0, size, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    if(data == MAP_FAILED)
    	return NULL;
    else
    	return data;
#endif
}

/* Frees a previously allocated data block
 *
 * @data  The pointer returned from a call to cphvb_memory_malloc
 * @size  The size of the allocated block
 * @return A pointer to data, and NULL on error
 */
cphvb_intp cphvb_memory_free(cphvb_data_ptr data, cphvb_intp size)
{
#ifdef _WIN32
	_aligned_free(data);
	return 0;
#else
	return munmap(data, size);
#endif
}
