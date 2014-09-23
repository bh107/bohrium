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

#include <pthread.h>
#include <unistd.h>
#include <stdint.h>
#include <stdint.h>
#include <sys/mman.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <map>
#include "bh_mem_signal.h"
#include <bh.h>

using namespace std;

static long PAGE_SIZE = sysconf(_SC_PAGESIZE);
static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
static bool initialized=false;

typedef struct
{
    //Id to identify the memory segment when executing the callback function.
    const void *idx;
    //Size of memory segment in bytes
    uint64_t size;
    //The callback function to call
    void (*callback)(void*, void*);
} segment;

static map<const void*,segment> segments;

//Return a iterator to the segment the 'addr' is part of.
//Returns segments.end() if 'addr' isn't in any segments
static map<const void*,segment>::const_iterator get_segment(const void *addr)
{
    map<const void*,segment>::const_iterator s = segments.lower_bound(addr);
    if(s != segments.end())
    {
        uint64_t offset = ((uint64_t)addr) - ((uint64_t)s->first);
        if(offset > s->second.size)
            return segments.end();
    }
    return s;
}

/** Signal handler.
 *  Executes appropriate callback function associated with memory segment.
 *
 * @param signal_number The signal number for SIGSEGV
 * @param siginfo_t Datastructure containing signal information.
 * @param context User context for the signal trap.
 */
static void sighandler(int signal_number, siginfo_t *info, void *context)
{
    pthread_mutex_lock(&mutex);
    map<const void*,segment>::const_iterator s = get_segment(info->si_addr);
    pthread_mutex_unlock(&mutex);
    if(s == segments.end())
    {
//        printf("bh_signal: Defaulting to segfaul at addr: %p\n", info->si_addr);
        signal(signal_number, SIG_DFL);
    }
    else
    {
        s->second.callback((void*)s->second.idx, info->si_addr);
    }
}

/** Init arrays and signal handler
 *
 * @param void
 * @returnm void
 */
int bh_mem_signal_init(void)
{
    pthread_mutex_lock(&mutex);
    if(!initialized)
    {
        struct sigaction sact;
        sigfillset(&(sact.sa_mask));
        sact.sa_flags = SA_SIGINFO | SA_ONSTACK;
        sact.sa_sigaction = sighandler;
        sigaction(SIGSEGV, &sact, &sact);
    }
    initialized = true;
    pthread_mutex_unlock(&mutex);
    return 0;
}

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
                         void (*callback)(void*, void*))
{
    pthread_mutex_lock(&mutex);
    if(get_segment(addr) != segments.end())
    {
        fprintf(stderr, "Could not attach signal, memory segment is in conflict with "
                        "already attached signal\n");
        return BH_ERROR;
    }
    segment &s = segments[addr];
    s.idx = idx;
    s.size = size;
    s.callback = callback;

    pthread_mutex_unlock(&mutex);
    return 0;
}

/** Detach signal
 *
 * @param addr - Start address of memory segment.
 * @return - error code
 */
int bh_mem_signal_detach(const void *addr)
{
    pthread_mutex_lock(&mutex);
    segments.erase(addr);
    pthread_mutex_unlock(&mutex);
    return 0;
}
