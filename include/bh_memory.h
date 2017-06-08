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

#ifndef __BH_MEMORY_H
#define __BH_MEMORY_H

#include <bh_type.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Allocate an alligned contigous block of memory,
 * without any initialization
 *
 * @size  The size of the allocated block
 * @return A pointer to data, and NULL on error
 */
void* bh_memory_malloc(int64_t size);

/* Frees a previously allocated data block
 *
 * @data  The pointer returned from a call to bh_memory_malloc
 * @size  The size of the allocated block
 * @return A pointer to data, and NULL on error
 */
int64_t bh_memory_free(void* data, int64_t size);

#ifdef __cplusplus
}
#endif

#endif

