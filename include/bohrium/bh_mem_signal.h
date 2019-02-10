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

#include <stdint.h>
#include <sys/mman.h>
#include <signal.h>

#ifdef __cplusplus
#include <sstream>
extern "C" {
#endif

/** Callback function type
 *  The function is called with the address pointer and the memory segment idx
 */
typedef int (*bh_mem_signal_callback_t) (void* fault_address, void* segment_idx);

/** Init arrays and signal handler
 *
 * @param void
 */
void bh_mem_signal_init(void);

/** Shutdown of this library */
void bh_mem_signal_shutdown(void);

/** Attach continues memory segment to signal handler
 *
 * @param idx - Id to identify the memory segment when executing the callback function.
 * @param addr - Start address of memory segment.
 * @param size - Size of memory segment in bytes
 * @param callback - Callback function which is executed when segfault hits in the memory
 *                   segment. The function is called with the address pointer and the memory segment idx.
 *                   NB: the function must return non-zero on success
 */
void bh_mem_signal_attach(void *idx, void *addr, uint64_t size, bh_mem_signal_callback_t callback);

/** Detach signal
 *
 * @param addr - Start address of memory segment.
 */
void bh_mem_signal_detach(const void *addr);

/** Check if signal exist
 *
 * @param addr - Start address of memory segment.
 */
int bh_mem_signal_exist(const void *addr);

/** Pretty print the segment data base
 *
 */
void bh_mem_signal_pprint_db(void);

#ifdef __cplusplus
}
#endif
