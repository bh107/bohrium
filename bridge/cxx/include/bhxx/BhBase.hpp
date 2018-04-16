/*
This file is part of Bohrium and copyright (c) 2012 the Bohrium team:
http://bohrium.bitbucket.org

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

#include <bh_view.hpp>
#include <bh_main_memory.hpp>
#include <memory>

namespace bhxx {
// The base underlying (multiple) arrays
class BhBase : public bh_base {
public:
    /** Is the memory managed referenced by bh_base's data pointer
     *  managed by Bohrium or is it owned externally
     *
     * \note If this flag is false, the class will make sure that
     *       the memory is not deleted when going out of scope.
     *  */
    bool ownMemory() {
        return m_own_memory;
    }

    /** Construct a base array with nelem elements using
     * externally managed storage.
     *
     * The class will make sure, that the storage is not deleted when
     * going out of scope.
     *
     * Needless to say that the memory should be large enough to
     * incorporate nelem_ elements.
     * */
    template <typename T>
    BhBase(size_t nelem_, T* memory) :
        m_own_memory(false) {
        data  = memory;
        nelem = static_cast<int64_t>(nelem_);
        set_type<T>();
    }

    /** Construct a base array and initialise it with the elements
     *  provided by an iterator range.
     *
     *  The values are copied into the Bohrium storage. If you want to
     *  provide external storage to Bohrium use the constructor
     *  BhBase(size_t nelem, T* memory) instead.
     */
    template <typename InputIterator, typename T = typename std::iterator_traits<InputIterator>::value_type>
    BhBase(InputIterator begin, InputIterator end)
          : BhBase(T(0), static_cast<size_t>(std::distance(begin, end))) {
        assert(std::distance(begin, end) > 0);

        // Allocate an array and copy the data over.
        bh_data_malloc(this);
        std::copy(begin, end, static_cast<T*>(data));
    }

    /** Construct a base array with nelem elements
     *
     * \param dummy   Dummy argument to fix the type of elements used.
     *                It may only have ever have the value 0 in the
     *                appropriate type.
     *
     * \note The use of this particular constructor is discouraged.
     *       It is only needed from BhArray to construct base objects
     *       which are uninitialised and do not yet hold any deta.
     *       If you wish to construct an uninitialised BhBase object,
     *       do this via the BhArray interface and not using this constructor.
     */
    template <typename T>
    BhBase(T dummy, size_t nelem_) :
        m_own_memory(true) {
        data  = nullptr;
        nelem = nelem_;
        set_type<T>();

        // The dummy is a dummy argument and should always be identical zero.
        assert(dummy == T(0));
    }

    ~BhBase() {
        // All memory here should be handed over to the Runtime
        // by a BH_FREE instruction and hence no BhBase object
        // should ever point to any memory on deletion
        assert(data == nullptr);
    }

    /** Deleted copy constructor */
    BhBase(const BhBase&) = delete;

    /** Deleted copy assignment */
    BhBase& operator=(const BhBase&) = delete;

    /** Delete move assignment */
    BhBase& operator=(BhBase&& other) = delete;
    // The reason for doing this is that there might still be
    // objects in the instruction queue which refer to the data
    // pointer used in this object. We hence cannot free the data,
    // but this is our only pointer to it within reach, so we
    // would theoretically need to free it here.

    /** Move another BhBase object here */
    BhBase(BhBase&& other) :
        bh_base(std::move(other)),
        m_own_memory(other.m_own_memory) {
        other.m_own_memory = true;
        other.data         = nullptr;
    }

private:
    /** Set the data type of the data pointed by data. */
    template <typename T>
    void set_type();

    // Is the memory in here owned by Bohrium or is it provided
    // by external means. If it is owned by Bohrium, we assume
    // that Bohrium has also allocated it, which means that if
    // someone is to extract this memory, then he will also
    // need to do the allocation using Bohrium's free function.
    //
    // This is assumed in the release_data() function and this is
    // why this filed is private.
    //
    // Also the BhBaseDeleter might stop working in quirky ways
    // if this flag is allowed to change randomly.
    bool m_own_memory;
};
}
