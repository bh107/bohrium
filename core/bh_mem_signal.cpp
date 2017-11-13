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
#include <cassert>
#include <stdexcept>
#include <set>
#include <iostream>
#include <sstream>

#include <bh_mem_signal.h>

using namespace std;

static pthread_mutex_t signal_mutex = PTHREAD_MUTEX_INITIALIZER;
static bool initialized = false;
static bool mem_warn = false;

struct Segment
{
    //Start address of this memory segment
    const void *addr;
    //Size of memory segment in bytes
    uint64_t size;
    //Id to identify the memory segment when executing the callback function.
    const void *idx;
    //The callback function to call
    void (*callback)(void*, void*);

    //Some constructors
    Segment(){};
    Segment(const void *addr) : addr(addr),size(1),idx(NULL),callback(NULL){};
    Segment(const void *addr, uint64_t size) : addr(addr),size(size),idx(NULL),callback(NULL){};
    Segment(const void *addr, uint64_t size, const void *idx, void (*callback)(void*, void*))
            : addr(addr),size(size),idx(idx),callback(callback){};

    bool operator<(const Segment& other) const
    {
        const uint64_t a_begin = (uint64_t) addr;
        const uint64_t b_begin = (uint64_t) other.addr;
        const uint64_t a_end = a_begin + size - 1;
        const uint64_t b_end = b_begin + other.size - 1;

        //When the two segments overlaps we return false such that
        //overlapping segments are identical in a set
        if((a_begin <= b_end) and (a_end >= b_begin))
        {
            return false;
        }
        else//Else we simple compare the begin address
            return a_begin < b_begin;
    }

    //Read begin and end memory address
    const void *addr_begin() const
    {
        return addr;
    }
    const void *addr_end() const
    {
        return (const void*)(((uint64_t)addr+size));
    }
};
//Pretty print of Segment
ostream& operator<<(ostream& out, const Segment& segment)
{
    out << segment.idx << "{addr: " << segment.addr_begin() << " - "
        << segment.addr_end() << "}";
    return out;
}
ostream& operator<<(ostream& out, const set<Segment>& segments)
{
    out << "bh_mem_signal contains: " << endl;
    for(const Segment &seg: segments)
    {
        out << seg << endl;
    }
    return out;
}

// All registered memory segments
// NB: never insert overlapping memory segments into this set
static set<Segment> segments;

/** Signal handler.
 *  Executes appropriate callback function associated with memory segment.
 *
 * @param signal_number The signal number for SIGSEGV
 * @param siginfo_t Datastructure containing signal information.
 * @param context User context for the signal trap.
 */
static void sighandler(int signal_number, siginfo_t *info, void *context)
{
    pthread_mutex_lock(&signal_mutex);
    set<Segment>::const_iterator s = segments.find(Segment(info->si_addr));
    pthread_mutex_unlock(&signal_mutex);
    if(s == segments.end())//Address not found in 'segments'
    {
        signal(signal_number, SIG_DFL);
    }
    else
    {
        s->callback((void*)s->idx, info->si_addr);
    }
}

void bh_mem_signal_init(void)
{
    mem_warn = getenv("BH_MEM_WARN") != NULL;

    pthread_mutex_lock(&signal_mutex);
    if(!initialized)
    {
        struct sigaction sact;
        sigfillset(&(sact.sa_mask));
        sact.sa_flags = SA_SIGINFO | SA_ONSTACK;
        sact.sa_sigaction = sighandler;
        sigaction(SIGSEGV, &sact, NULL);
        sigaction(SIGBUS, &sact, NULL);
    }
    initialized = true;
    pthread_mutex_unlock(&signal_mutex);
}

void bh_mem_signal_shutdown(void)
{
    pthread_mutex_lock(&signal_mutex);
    if(segments.size() > 0) {
        if (mem_warn) {
            cout << "MEM_WARN: bh_mem_signal_shutdown() - not all attached memory segments are detached!" << endl;
            bh_mem_signal_pprint_db();
        }
    }
    pthread_mutex_unlock(&signal_mutex);
}

void bh_mem_signal_attach(const void *idx, const void *addr, uint64_t size,
                          void (*callback)(void*, void*))
{
    pthread_mutex_lock(&signal_mutex);

    // Create new memory segment that we will attach
    Segment segment(addr, size, idx, callback);

    // Let's check for double attachments
    if(segments.find(segment) != segments.end())
    {
        auto conflict = segments.find(Segment(addr, size));
        stringstream ss;
        ss << "mem_signal: Could not attach signal, memory segment (" \
           << segment.addr_begin() << " to " << segment.addr_end() \
           << ") is in conflict with already attached memory segment (" \
           << conflict->addr_begin() << " to " << conflict->addr_end() << ")" << endl;
        pthread_mutex_unlock(&signal_mutex);
        throw runtime_error(ss.str());
    }

    // Finally, let's insert the new segment
    segments.insert(segment);
    pthread_mutex_unlock(&signal_mutex);
}

void bh_mem_signal_detach(const void *addr)
{
    pthread_mutex_lock(&signal_mutex);
    segments.erase(addr);
    pthread_mutex_unlock(&signal_mutex);
}

int bh_mem_signal_exist(const void *addr)
{
    int ret;
    pthread_mutex_lock(&signal_mutex);
    ret = segments.find(addr) != segments.end();
    pthread_mutex_unlock(&signal_mutex);
    return ret;
}

void bh_mem_signal_pprint_db(void)
{
    cout << segments << endl;
}
