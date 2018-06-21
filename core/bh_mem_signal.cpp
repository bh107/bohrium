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
#include <stdio.h>
#include <stdlib.h>
#include <cassert>
#include <stdexcept>
#include <set>
#include <iostream>
#include <sigsegv.h>
#include <bh_mem_signal.h>
#include <bh_util.hpp>

using namespace std;

static pthread_mutex_t signal_mutex = PTHREAD_MUTEX_INITIALIZER;
static bool initialized = false;
static bool mem_warn = false;

struct Segment {
    //Start address of this memory segment
    const void *addr;
    //Size of memory segment in bytes
    uint64_t size;
    //Id to identify the memory segment when executing the callback function.
    const void *idx;
    //The callback function to call
    bh_mem_signal_callback_t callback;
    //sigsegv ticket of a registered memory range
    void *ticket;

    //Some constructors
    Segment() {};

    Segment(const void *addr) : addr(addr), size(1), idx(nullptr), callback(nullptr), ticket(nullptr) {};

    Segment(const void *addr, uint64_t size) : addr(addr), size(size), idx(nullptr),
                                               callback(nullptr), ticket(nullptr) {};

    Segment(const void *addr, uint64_t size, const void *idx) : addr(addr), size(size), idx(idx),
                                                                callback(nullptr), ticket(nullptr) {};

    void add_callback_and_ticket(bh_mem_signal_callback_t callback, void *ticket) {
        this->callback = callback;
        this->ticket = ticket;
    }

    bool operator<(const Segment &other) const {
        const uint64_t a_begin = (uint64_t) addr;
        const uint64_t b_begin = (uint64_t) other.addr;
        const uint64_t a_end = a_begin + size - 1;
        const uint64_t b_end = b_begin + other.size - 1;

        //When the two segments overlaps we return false such that
        //overlapping segments are identical in a set
        if ((a_begin <= b_end) and (a_end >= b_begin)) {
            return false;
        } else { //Else we simple compare the begin address
            return a_begin < b_begin;
        }
    }

    //Read begin and end memory address
    const void *addr_begin() const {
        return addr;
    }

    const void *addr_end() const {
        return (const void *) (((uint64_t) addr + size));
    }
};

//Pretty print of Segment
ostream &operator<<(ostream &out, const Segment &segment) {
    out << segment.idx << "{addr: " << segment.addr_begin() << " - "
        << segment.addr_end() << ", ticket: " << segment.ticket << "}";
    return out;
}

ostream &operator<<(ostream &out, const set<Segment> &segments) {
    out << "bh_mem_signal contains: " << endl;
    for (const Segment &seg: segments) {
        out << seg << endl;
    }
    return out;
}

// sigsegv boilerplate
static sigsegv_dispatcher dispatcher;
static int handler(void *fault_address, int serious) {
    // We only handle serious faults and not potential faults such as stack overflows
    if (serious == 1) {
        return sigsegv_dispatch(&dispatcher, fault_address);
    } else {
        return 0;
    }
}

// All registered memory segments
// NB: never insert overlapping memory segments into this set
static set<Segment> segments;

void bh_mem_signal_init(void) {
    mem_warn = getenv("BH_MEM_WARN") != nullptr;

    pthread_mutex_lock(&signal_mutex);
    if (!initialized) {
        sigsegv_init (&dispatcher);
        if (sigsegv_install_handler(&handler) == -1) {
            throw runtime_error("System cannot catch SIGSEGV");
        }
    }
    initialized = true;
    pthread_mutex_unlock(&signal_mutex);
}

void bh_mem_signal_shutdown(void) {
    pthread_mutex_lock(&signal_mutex);
    if (not segments.empty()) {
        if (mem_warn) {
            cout << "MEM_WARN: bh_mem_signal_shutdown() - not all attached memory segments are detached!" << endl;
            bh_mem_signal_pprint_db();
        }
    }
    if (initialized) {
        sigsegv_deinstall_handler();
    }
    pthread_mutex_unlock(&signal_mutex);
}

void bh_mem_signal_attach(void *idx, void *addr, uint64_t size, bh_mem_signal_callback_t callback) {
    pthread_mutex_lock(&signal_mutex);

    // Create new memory segment that we will attach
    Segment segment(addr, size, idx);

    // Let's check for double attachments
    if (util::exist(segments, segment)) {
        auto conflict = segments.find(Segment(addr, size));
        stringstream ss;
        ss << "mem_signal: Could not attach signal, memory segment (" \
           << segment.addr_begin() << " to " << segment.addr_end() \
           << ") is in conflict with already attached memory segment (" \
           << conflict->addr_begin() << " to " << conflict->addr_end() << ")" << endl;
        pthread_mutex_unlock(&signal_mutex);
        throw runtime_error(ss.str());
    }
#ifdef SIGSEGV_FAULT_ADDRESS_ALIGNMENT // SIGSEGV_FAULT_ADDRESS_ALIGNMENT isn't defined in older versions
    assert(((size_t) addr) % SIGSEGV_FAULT_ADDRESS_ALIGNMENT == 0);
    assert(size % SIGSEGV_FAULT_ADDRESS_ALIGNMENT == 0);
#endif
    // Let's register it in sigsegv and save it in the segment
    void *ticket = sigsegv_register(&dispatcher, addr, size, callback, idx);
    segment.add_callback_and_ticket(callback, ticket);

    // Finally, let's insert the new segment
    segments.insert(segment);
    pthread_mutex_unlock(&signal_mutex);
}

void bh_mem_signal_detach(const void *addr) {
    pthread_mutex_lock(&signal_mutex);
    auto it = segments.find(addr);
    if (it != segments.end()) {
        assert(it->ticket != nullptr);
        sigsegv_unregister(&dispatcher, it->ticket);
        segments.erase(it);
    }
    pthread_mutex_unlock(&signal_mutex);
}

int bh_mem_signal_exist(const void *addr) {
    int ret;
    pthread_mutex_lock(&signal_mutex);
    ret = util::exist(segments, addr);
    pthread_mutex_unlock(&signal_mutex);
    return ret;
}

void bh_mem_signal_pprint_db(void) {
    cout << segments << endl;
}
