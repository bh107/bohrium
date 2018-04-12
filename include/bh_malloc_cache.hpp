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

#include <vector>
#include <sstream>
#include <stdexcept>
#include <sys/mman.h>
#include <bh_util.hpp>

namespace bohrium {

class MallocCache {
private:
    struct Segment {
        std::size_t nbytes;
        void *mem;
    };
    std::vector<Segment> _segments;
    size_t _total_num_bytes = 0;
    size_t _total_num_lookups = 0;
    size_t _total_num_misses = 0;
    size_t _total_mem_allocated = 0;
    size_t _max_mem_allocated = 0;

    static constexpr size_t MAX_NBYTES = 1000000;

    void *_malloc(size_t nbytes) {
        // Allocate page-size aligned memory.
        // The MAP_PRIVATE and MAP_ANONYMOUS flags is not 100% portable. See:
        // <http://stackoverflow.com/questions/4779188/how-to-use-mmap-to-allocate-a-memory-in-heap>
        void *ret = mmap(0, nbytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (ret == MAP_FAILED or ret == nullptr) {
            std::stringstream ss;
            ss << "MallocCache() could not allocate a data region. Returned error code: " << strerror(errno);
            throw std::runtime_error(ss.str());
        }
        _total_mem_allocated += nbytes;
        if (_total_mem_allocated > _max_mem_allocated) {
            _max_mem_allocated = _total_mem_allocated;
        }
//        std::cout << "malloc       - nbytes: " << nbytes << ",  addr: " << ret << std::endl;
        return ret;
    }

    void _free(void *mem, size_t nbytes) {
//        std::cout << "free         - nbytes: " << nbytes << ",  addr: " << mem << std::endl;
        assert(mem != nullptr);
        if (munmap(mem, nbytes) != 0) {
            std::stringstream ss;
            ss << "MallocCache() could not free a data region. " << "Returned error code: " << strerror(errno);
            throw std::runtime_error(ss.str());
        }
        assert(_max_mem_allocated >= nbytes);
        _total_mem_allocated -= nbytes;
    }

    void _erase(std::vector<Segment>::iterator first,
                std::vector<Segment>::iterator last, bool call_free) {
        for (auto it = first; it != last; ++it) {
            if (call_free) {
                _free(it->mem, it->nbytes);
            }
            _total_num_bytes -= it->nbytes;
        }
        _segments.erase(first, last);
    }

    void _erase(std::vector<Segment>::iterator position, bool call_free) {
        _erase(position, std::next(position), call_free);
    }

    void _erase(std::vector<Segment>::reverse_iterator position, bool call_free) {
        // Notice, we need to iterate `position` once when converting from reverse to regular iterator
        _erase(std::next(position).base(), call_free);
    }

public:

    size_t shrink(size_t nbytes) {
        size_t count = 0;
        std::vector<Segment>::iterator it;
        for (it = _segments.begin(); it != _segments.end() and count < nbytes; ++it) {
            count += it->nbytes;
        }
        _erase(_segments.begin(), it, true);
        return count;
    }

    size_t shrinkToFit(size_t total_num_bytes) {
        if (total_num_bytes < _total_num_bytes) {
            shrink(_total_num_bytes - total_num_bytes);
        }
        return _total_num_bytes;
    }

    void *alloc(size_t nbytes) {
        if (nbytes == 0) {
            return nullptr;
        }
        ++_total_num_lookups;
        // Check for segment of size `nbytes`, which is a cache hit!
        for (auto it = _segments.rbegin(); it != _segments.rend(); ++it) { // Search in reverse
            if (it->nbytes == nbytes) {
                void *ret = it->mem;
                assert(ret != nullptr);
                _erase(it, false);
    //            std::cout << "cache hit!   - nbytes: " << nbytes << ",  addr: " << ret << std::endl;
                return ret;
            }
        }
        ++_total_num_misses;
        void *ret = _malloc(nbytes); // Cache miss
    //    std::cout << "cache miss!  - nbytes: " << nbytes << ",  addr: " << ret << std::endl;
        return ret;
    }

    void free(size_t nbytes, void *memory) {
        // Let's make sure that we don't exceed `MAX_NBYTES`
        if (nbytes > MAX_NBYTES) {
            return _free(memory, nbytes);
        }
        shrinkToFit(MAX_NBYTES - nbytes);

        // Insert the segment at the end of `_segments`
        Segment seg;
        seg.nbytes = nbytes;
        seg.mem = memory;
        _segments.push_back(seg);
        _total_num_bytes += nbytes;
//        std::cout << "cache insert - nbytes: " << nbytes << ",  addr: " << seg.mem << std::endl;
    }

    ~MallocCache() {
        shrinkToFit(0);
        assert(_total_mem_allocated == 0);
    }

    size_t getTotalNumBytes() const {
        return _total_num_bytes;
    }

    size_t getTotalNumLookups() const {
        return _total_num_lookups;
    }

    size_t getTotalNumMisses() const {
        return _total_num_misses;
    }

    size_t getMaxMemAllocated() const {
        return _max_mem_allocated;
    }

};


} // Namespace bohrium
