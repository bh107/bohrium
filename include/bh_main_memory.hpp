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

#include <cstddef>
#include <bh_base.hpp>

/** Return the size of the physical memory on this machine */
uint64_t bh_main_memory_total();

/** Allocate data memory for the given base if not already allocated.
 * For convenience, the base is allowed to be NULL.
 *
 * @base    The base in question
 */
void bh_data_malloc(bh_base* base);

/** Frees data memory for the given view.
 * For convenience, the view is allowed to be NULL.
 *
 * @base    The base in question
 */
void bh_data_free(bh_base* base);

/** Set the size limit of the main memory malloc cache (see MallocCache::setLimit())
 *
 * @param nbytes The memory limit in bytes
 */
void bh_set_malloc_cache_limit(uint64_t nbytes);

/** Retrieve statistic from the main memory malloc cache
 *
 * @param cache_lookup Cache lookups
 * @param cache_misses Cache misses
 * @param max_memory_usage Total memory usage, which includes ALL memory allocated through the memory cache
 */
void bh_get_malloc_cache_stat(uint64_t &cache_lookup, uint64_t &cache_misses, uint64_t &max_memory_usage);
