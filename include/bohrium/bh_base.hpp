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

#include <bohrium/bh_type.hpp>
#include <bohrium/bh_constant.hpp>

#include <boost/serialization/split_member.hpp>

// Forward declaration of class boost::serialization::access
namespace boost { namespace serialization { class access; }}

/** This class represent a data distribution layout.
 *  For now, we simply distribute over the flatten base where each process gets one data block.
 */
class BhPGAS {
    // Is pgas enabled?
    bool _enabled = false;
    // The number of elements in the array globally
    int64_t _global_size = 0;
    // The total number of processes
    int _comm_size = 0;
    // The rank of the current process
    int _comm_rank = 0;

public:

    BhPGAS() = default;

    /** Construct a disabled PGAS object for regular non-distributed arrays
     *
     * @param nelem Number of elements in the regular non-distributed array
     */
    explicit BhPGAS(int64_t nelem) : _global_size{nelem} {}

    explicit BhPGAS(int64_t global_size, int comm_size, int comm_rank) : _enabled{true},
                                                                         _global_size{global_size},
                                                                         _comm_size{comm_size},
                                                                         _comm_rank{comm_rank} {}

    /// Return true when pgas is enabled
    bool enabled() const {
        return _enabled;
    }

    /// Return the total number of communicating processes
    int commSize() const {
        return _comm_size;
    }

    /// Return the rank of this current communicating process
    int commRank() const {
        return _comm_rank;
    }

    /// Return the global size of the array
    int64_t globalSize() const {
        return _global_size;
    }

    /// Return the local size of the array (if disabled, `localSize() == globalSize()`)
    int64_t localSize() const {
        int64_t local_size = globalSize();
        if (enabled()) {
            local_size = globalSize() / commSize();
            if (commRank() == commSize() - 1) {
                local_size += globalSize() % commSize();
            }
        }
        return local_size;
    }

    /// Serialization using Boost
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive &ar, const unsigned int version) {
        ar & _enabled;
        ar & _global_size;
        ar & _comm_size;
        ar & _comm_rank;
    }
};


/** The base underlying (multiple) arrays */
class bh_base {
private:
    // The number of elements in the array
    int64_t _nelem = 0;

    // The type of data in the array
    bh_type _type = bh_type::BOOL;

    // Pointer to the actual data.
    void *_data = nullptr;

public:

    // The pgas object
    BhPGAS pgas;

    bh_base() = default;

    /** Construct a new base array
     *
     * @param nelem  Number of elements
     * @param type   The data type
     * @param data   Pointer to the actual data (or nullptr)
     */
    bh_base(int64_t nelem, bh_type type, void *data = nullptr) : _nelem(nelem), _type(type), _data(data), pgas{nelem} {}

    bh_base(int64_t nelem, bh_type type, BhPGAS pgas) : _nelem(nelem), _type(type), pgas{std::move(pgas)} {}

    /// Returns the number of elements in this array
    int64_t nelem() const noexcept {
        return _nelem;
    }

    /// Returns the data type of this array
    bh_type dtype() const noexcept {
        return _type;
    }

    /// Returns the data pointer of this array
    void *getDataPtr() {
        return _data;
    }

    /// Returns the data pointer of this array (const version)
    const void *getDataPtr() const {
        return _data;
    }

    /// Reset the data pointer for this array to `data_ptr`
    void resetDataPtr(void *data_ptr = nullptr) {
        _data = data_ptr;
    }

    /// Returns an unique ID of this base array
    uint64_t getLabel() const;

    /// Returns pprint string of this base array
    std::string str() const;

    /// Returns the of bytes in the array
    int64_t nbytes() const {
        return nelem() * bh_type_size(_type);
    };

    template<class Archive>
    void save(Archive &ar, const unsigned int version) const {
        size_t tmp = (size_t) _data;
        ar << tmp;
        ar & _type;
        ar & _nelem;
        ar & pgas;
    }

    template<class Archive>
    void load(Archive &ar, const unsigned int version) {
        size_t tmp;
        ar >> tmp;
        _data = (void *) tmp;
        ar & _type;
        ar & _nelem;
        ar & pgas;
    }

    BOOST_SERIALIZATION_SPLIT_MEMBER()
};
