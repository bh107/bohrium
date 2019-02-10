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

struct bh_base {
private:
    // The number of elements in the array
    int64_t _nelem = 0;

    // The type of data in the array
    bh_type _type = bh_type::BOOL;

    // Pointer to the actual data.
    void *_data = nullptr;

public:
    bh_base() = default;

    /** Construct a new base array
     *
     * @param nelem  Number of elements
     * @param type   The data type
     * @param data   Pointer to the actual data (or nullptr)
     */
    bh_base(int64_t nelem, bh_type type, void* data = nullptr) : _nelem(nelem), _type(type), _data(data) {}

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
    }

    template<class Archive>
    void load(Archive &ar, const unsigned int version) {
        size_t tmp;
        ar >> tmp;
        _data = (void *) tmp;
        ar & _type;
        ar & _nelem;
    }

    BOOST_SERIALIZATION_SPLIT_MEMBER()
};
