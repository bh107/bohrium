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

#ifndef __BH_MEM_SIGNAL_H
#define __BH_MEM_SIGNAL_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <sys/mman.h>
#include <signal.h>

/** Init arrays and signal handler
 *
 * @param void
 * @returnm void
 */
int bh_mem_signal_init(void);

/** Attach continues memory segment to signal handler
 *
 * @param idx - Id to identify the memory segment when executing the callback function.
 * @param addr - Start address of memory segment.
 * @param size - Size of memory segment in bytes
 * @param callback - Callback function which is executed when segfault hits in the memory
 *                   segment. The function is called with the memory idx and the address pointer
 * @return - error code
 */
int bh_mem_signal_attach(const void *idx, const void *addr, uint64_t size,
                         void (*callback)(void*, void*));

/** Detach signal
 *
 * @param addr - Start address of memory segment.
 * @return - error code
 */
int bh_mem_signal_detach(const void *addr);

#ifdef __cplusplus
}
#endif

#endif
