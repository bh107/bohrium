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

#include "bh_type.hpp"
#include <bh_constant.hpp>

#include <boost/serialization/split_member.hpp>

// Forward declaration of class boost::serialization::access
namespace boost {namespace serialization {class access;}}

struct bh_base
{
    // Pointer to the actual data.
    void*   data;

    // The type of data in the array
    bh_type       type;

    // The number of elements in the array
    int64_t      nelem;

    // Returns an unique ID of this base array
    size_t get_label() const;

    // Returns pprint string of this base array
    std::string str() const;

    // Returns the of bytes in the array
    size_t nbytes() const {
        return nelem * bh_type_size(type);
    };

    template<class Archive>
    void save(Archive & ar, const unsigned int version) const
    {
        size_t tmp = (size_t)data;
        ar << tmp;
        ar & type;
        ar & nelem;
    }
    template<class Archive>
    void load(Archive & ar, const unsigned int version)
    {
        size_t tmp;
        ar >> tmp;
        data = (void*)tmp;
        ar & type;
        ar & nelem;
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()
};
