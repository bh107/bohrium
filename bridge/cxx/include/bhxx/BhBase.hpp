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
    bool own_memory() { return m_own_memory; }

    /** Construct a base array with nelem elements */
    BhBase(size_t nelem_) : m_own_memory(true) {
        data  = nullptr;
        nelem = static_cast<bh_index>(nelem_);
    }

    /** Set the data type of the data pointed by data. */
    template <typename T>
    void set_type();

    /** Construct a base array with nelem elements using
     * externally managed storage.
     *
     * The class will make sure, that the storage is not deleted when
     * going out of scope.
     * */
    template <typename T>
    BhBase(size_t nelem_, T* memory) : m_own_memory(false) {
        data  = memory;
        nelem = static_cast<bh_index>(nelem_);
        set_type<T>();
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

    /** Move another BhBase object here */
    BhBase(BhBase&& other) : bh_base(std::move(other)), m_own_memory(other.m_own_memory) {
        other.m_own_memory = true;
        other.data         = nullptr;
    }

    /** Move-assign a BhBase object */
    BhBase& operator=(BhBase&&) = delete;
    // TODO Implement the upper guy

  private:
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
