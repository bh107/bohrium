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

#pragma once

#define NO_IMPORT_ARRAY
#define NO_IMPORT_BH_API
#include "_bh.h"

/** Memory map - allocate aligned memory
 *
 * @param nbytes  Number of bytes to allocate
 * @return        Pointer to the new memory
 */
void* mem_map(uint64_t nbytes);

/** Memory un-map
 *
 * @param addr Memory address
 * @param size Memory size
 */
void mem_unmap(void *addr, npy_intp size);

/** Allocate protected memory for the NumPy part of 'ary'
 * This function only allocates if the 'ary' is a new base array and avoids multiple
 * allocations by checking and setting the `ary->mmap_allocated`
 * @param ary The array in question
 */
void protected_malloc(BhArray *ary);

/** Attach signal handler to the memory
 *
 * @param idx  Id to identify the memory segment when executing `mem_access_callback()`.
 * @param addr Start address of memory segment.
 * @param size Size of memory segment in bytes
 */
void mem_signal_attach(void *idx, void *addr, uint64_t nbytes);

/** Move data from the bhc domain to the NumPy domain
 *
 * @param base_array The base array in question
 */
void mem_bhc2np(BhArray *base_array);

/** Move data from the NumPy domain to the bhc domain
 *
 * @param base_array The base array in question
 */
void mem_np2bhc(BhArray *base_array);