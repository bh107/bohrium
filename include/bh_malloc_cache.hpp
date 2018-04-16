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
#include <bh_util.hpp>

namespace bohrium {

/** Cache of memory allocations. Instead of freeing a memory allocation immediately, this cache
 * retain the allocation for later reuse.
 * To use, simply allocate and free all memory allocations through the method `alloc()` and `free()`
 */
class MallocCache {
public:
    typedef std::function<void *(uint64_t)> FuncAllocT;
    typedef std::function<void(void *, uint64_t)> FuncFreeT;

private:
    // A segment consist of a memory allocation and a size
    struct Segment {
        std::uint64_t nbytes;
        void *mem;
    };
    std::vector<Segment> _segments; // Segments in the cache

    // Pointers to malloc and free functions
    FuncAllocT _func_alloc;
    FuncFreeT _func_free;

    uint64_t _cache_size = 0; // Current size of the cache (in bytes)
    uint64_t _mem_allocated = 0; // Current memory allocated inside and outside the cache (in bytes)
    uint64_t _mem_allocated_limit; // The limit of `_mem_allocated`

    // Some statistics
    uint64_t _stat_lookups = 0;
    uint64_t _stat_misses = 0;
    uint64_t _stat_allocated_max = 0;

    /** Allocate memory of size `nbytes`
     *
     * @param nbytes Number of bytes to allocate
     * @return Pointer to the allocation
     */
    void *_malloc(uint64_t nbytes) {
        void *ret = _func_alloc(nbytes);
        _mem_allocated += nbytes;
        if (_mem_allocated > _stat_allocated_max) {
            _stat_allocated_max = _mem_allocated;
        }
        return ret;
    }

    /** Free the memory allocation `mem` of size `nbytes` */
    void _free(void *mem, uint64_t nbytes) {
        assert(mem != nullptr);
        _func_free(mem, nbytes);
        assert(_mem_allocated >= nbytes);
        _mem_allocated -= nbytes;
    }

    /** Evict a range of memory allocations from the cache
     *
     * @param first Iterator pointing to the first allocation
     * @param last Iterator pointing to the element just after the last allocation
     * @param call_free When true, the memory allocations are also freed
     */
    void _evict(std::vector<Segment>::iterator first, std::vector<Segment>::iterator last, bool call_free) {
        for (auto it = first; it != last; ++it) {
            if (call_free) {
                _free(it->mem, it->nbytes);
            }
            _cache_size -= it->nbytes;
        }
        _segments.erase(first, last);
    }

    /** Evict a memory allocation from the cache
     *
     * @param position Iterator pointing to the allocation
     * @param call_free When true, the memory allocations are also freed
     */
    void _evict(std::vector<Segment>::iterator position, bool call_free) {
        _evict(position, std::next(position), call_free);
    }

    /** Evict a memory allocation from the cache
     * 
     * @param position Reverse iterator pointing to the allocation
     * @param call_free When true, the memory allocations are also freed
     */
    void _evict(std::vector<Segment>::reverse_iterator position, bool call_free) {
        // Notice, we need to iterate `position` once when converting from reverse to regular iterator
        _evict(std::next(position).base(), call_free);
    }

public:

    /** Constructor
     *
     * @param func_alloc A function that takes size and returns a new memory allocation
     * @param func_free  A function that takes a memory allocation and size and frees the allocation
     * @param limit_num_bytes The size limit of the cache (see setLimit())
     */
    MallocCache(FuncAllocT func_alloc, FuncFreeT func_free, uint64_t limit_num_bytes) :
            _func_alloc(func_alloc), _func_free(func_free), _mem_allocated_limit(limit_num_bytes) {}

    /** Pretty print the cache */
    std::string pprint() {
        std::stringstream ss;
        ss << "Malloc Cache: \n";
        for (const Segment &seg: _segments) {
            ss << "  (" << seg.nbytes << "B, " << seg.mem << ")\n";
        }
        return ss.str();
    }

    /** Shrink to size of the cache with at least `nbytes`
     *
     * @param nbytes The minimum amount of bytes to shrink with
     * @return The actual size reduction
     */
    uint64_t shrink(uint64_t nbytes) {
        uint64_t count = 0;
        std::vector<Segment>::iterator it;
        for (it = _segments.begin(); it != _segments.end() and count < nbytes; ++it) {
            count += it->nbytes;
        }
        _evict(_segments.begin(), it, true);
        return count;
    }

    /** Makes sure that the size of the cache is at most `nbytes`
     *
     * @param nbytes The maximum size of the cache size
     * @return The size reduction (if any)
     */
    uint64_t shrinkToFit(uint64_t nbytes) {
        if (nbytes < _cache_size) {
            return shrink(_cache_size - nbytes);
        }
        return 0;
    }

    /** Shrink the cache to fit in the `_mem_allocated_limit` limit
     *
     * @param extra_mem_allocated  Additional number of bytes added to `_mem_allocated` before checking for overflow
     */
    void shrinkToFitLimit(uint64_t extra_mem_allocated = 0) {
        const uint64_t mem_alloc = _mem_allocated + extra_mem_allocated;
        if (mem_alloc > _mem_allocated_limit) { // We are above the limit
            assert(mem_alloc >= _cache_size);
            const uint64_t mem_not_in_cache = mem_alloc - _cache_size;
            if (mem_not_in_cache < _mem_allocated_limit) {
                shrinkToFit(_mem_allocated_limit - mem_not_in_cache);
            } else {
                shrinkToFit(0); // We can at most empty the cache
            }
        }
    }

    /** Alloc a memory allocation of size `nbytes`
     *
     * @param nbytes Number of bytes to allocate
     * @return The memory allocation
     */
    void *alloc(uint64_t nbytes) {
        if (nbytes == 0) {
            return nullptr;
        }
        ++_stat_lookups;
        // Check for segment of size `nbytes`, which is a cache hit!
        for (auto it = _segments.rbegin(); it != _segments.rend(); ++it) { // Search in reverse
            if (it->nbytes == nbytes) {
                void *ret = it->mem;
                assert(ret != nullptr);
                _evict(it, false);
                return ret;
            }
        }
        ++_stat_misses;

        // Since we are allocating new memory, we might have to shrink to fit `_mem_allocated_limit`
        shrinkToFitLimit(nbytes);

        void *ret = _malloc(nbytes); // Cache miss
        return ret;
    }

    /** Frees a memory allocation of size `nbytes`
     *
     * @param nbytes The size of the memory allocation
     * @param memory The memory allocation
     */
    void free(uint64_t nbytes, void *memory) {
        if (_mem_allocated_limit == 0) {
            _free(memory, nbytes);
        } else {
            // Insert the segment at the end of `_segments`
            Segment seg;
            seg.nbytes = nbytes;
            seg.mem = memory;
            _segments.push_back(seg);
            _cache_size += nbytes;
        }
    }

    /** Destructor */
    ~MallocCache() {
        shrinkToFit(0);
        assert(_cache_size == 0);
    }

    /** Set the size limit of this cache. The limit is in terms of all memory allocated, which include both
     * allocations inside and outside the cache.
     * NB: the total amount of allocated memory might exceed the limit since we can at most shrink the cache to zero.
     *
     * @param nbytes The limit in bytes
     */
    void setLimit(uint64_t nbytes) {
        _mem_allocated_limit = nbytes;
        shrinkToFitLimit();
    };

    uint64_t getTotalNumBytes() const {
        return _cache_size;
    }

    uint64_t getTotalNumLookups() const {
        return _stat_lookups;
    }

    uint64_t getTotalNumMisses() const {
        return _stat_misses;
    }

    uint64_t getMaxMemAllocated() const {
        return _stat_allocated_max;
    }
};


} // Namespace bohrium
