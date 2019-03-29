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

#include <bohrium/bh_main_memory.hpp>
#include <bohrium/bh_malloc_cache.hpp>
#include <bohrium/jitk/subprocess.hpp>
#include <sys/mman.h>
#include <sys/types.h>
#include <boost/regex.hpp>

#if defined(__APPLE__) || defined(__MACOSX)
#include <sys/sysctl.h>
#else

#include <sys/sysinfo.h>

#endif

using namespace std;
using namespace bohrium;
namespace P = subprocess;

namespace {
int64_t _run_command_and_grab_integer(const string &cmd, const string &regex) {
    P::Popen p = P::Popen(cmd, P::output{P::PIPE}, P::error{P::PIPE});
    auto res = p.communicate();
    std::string std_out(res.first.buf.begin(), res.first.buf.end()); // Stdout
    std::string std_err(res.second.buf.begin(), res.second.buf.end()); // Stderr

    if (p.retcode() > 0) {
        return -1;
    }
    const boost::regex expr{regex};
    boost::smatch match;
    if (!boost::regex_search(std_out, match, expr) || match.size() < 2) {
        return -1;
    }
    return std::stoll(match[1].str());
}
}

uint64_t bh_main_memory_total() {
#if defined(__APPLE__) || defined(__MACOSX)
    int mib[2];
    int64_t physical_memory;
    mib[0] = CTL_HW;
    mib[1] = HW_MEMSIZE;
    size_t length = sizeof(int64_t);
    sysctl(mib, 2, &physical_memory, &length, nullptr, 0);
    return physical_memory;
#else
    struct sysinfo memInfo;
    sysinfo(&memInfo);
    return memInfo.totalram * memInfo.mem_unit;
#endif
}

int64_t bh_main_memory_unused() {
    #if defined(__APPLE__) || defined(__MACOSX)
    return _run_command_and_grab_integer("top -l1", ",\\s*(\\d+)\\s*M unused") * 1024 * 1024;
    #else
    return _run_command_and_grab_integer("cat /proc/meminfo", "MemAvailable:\\s+(\\d+)\\s*kB") * 1024;
    #endif
}


namespace {
// Allocate page-size aligned main memory.
void *main_mem_malloc(uint64_t nbytes) {
    // The MAP_PRIVATE and MAP_ANONYMOUS flags is not 100% portable. See:
    // <http://stackoverflow.com/questions/4779188/how-to-use-mmap-to-allocate-a-memory-in-heap>
    void *ret = mmap(0, nbytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (ret == MAP_FAILED or ret == nullptr) {
        std::stringstream ss;
        ss << "main_mem_malloc() could not allocate a data region. Returned error code: " << strerror(errno);
        throw std::runtime_error(ss.str());
    }
    return ret;
}

void main_mem_free(void *mem, uint64_t nbytes) {
    assert(mem != nullptr);
    if (munmap(mem, nbytes) != 0) {
        std::stringstream ss;
        ss << "main_mem_free() could not free a data region. " << "Returned error code: " << strerror(errno);
        throw std::runtime_error(ss.str());
    }
}

MallocCache malloc_cache(main_mem_malloc, main_mem_free, 0);
}

void bh_data_malloc(bh_base *base) {
    if (base == nullptr) return;
    if (base->getDataPtr() != nullptr) return;
    base->resetDataPtr(malloc_cache.alloc(base->nbytes()));
}

void bh_data_free(bh_base *base) {
    if (base == nullptr) return;
    if (base->getDataPtr() == nullptr) return;
    malloc_cache.free(base->nbytes(), base->getDataPtr());
    base->resetDataPtr();
}

void bh_set_malloc_cache_limit(uint64_t nbytes) {
    malloc_cache.setLimit(nbytes);
}

void bh_get_malloc_cache_stat(uint64_t &cache_lookup, uint64_t &cache_misses, uint64_t &max_memory_usage) {
    cache_lookup = malloc_cache.getTotalNumLookups();
    cache_misses = malloc_cache.getTotalNumMisses();
    max_memory_usage = malloc_cache.getMaxMemAllocated();
}

