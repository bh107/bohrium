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

#ifndef __BH_ARRAY_H
#define __BH_ARRAY_H

#include <algorithm>
#include <stdbool.h>
#include <iostream>
#include "bh_type.h"
#include "bh_win.h"

#include <boost/serialization/split_member.hpp>

// Forward declaration of class boost::serialization::access
namespace boost {namespace serialization {class access;}}

#define BH_MAXDIM (16)

struct bh_base
{
    /// Pointer to the actual data.
    bh_data_ptr   data;

    /// The type of data in the array
    bh_type       type;

    /// The number of elements in the array
    bh_index      nelem;

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

struct bh_view
{
    /// Pointer to the base array.
    bh_base*      base;

    /// Index of the start element
    bh_index      start;

    /// Number of dimensions
    bh_intp       ndim;

    /// Number of elements in each dimensions
    bh_index      shape[BH_MAXDIM];

    /// The stride for each dimensions
    bh_index      stride[BH_MAXDIM];

    bool operator<(const bh_view& other) const
    {
        if (base < other.base) return true;
        if (other.base < base) return false;
        if (start < other.start) return true;
        if (other.start < start) return false;
        if (ndim < other.ndim) return true;
        if (other.ndim < ndim) return false;
        for (bh_intp i = 0; i < ndim; ++i)
        {
            if (shape[i] < other.shape[i]) return true;
            if (other.shape[i] < shape[i]) return false;
        }
        for (bh_intp i = 0; i < ndim; ++i)
        {
            if (stride[i] < other.stride[i]) return true;
            if (other.stride[i] < stride[i]) return false;
        }
        return false;
    }
    bool operator==(const bh_view& other) const
    {
        if (base != other.base) return false;
        if (ndim != other.ndim) return false;
        if (start != other.start) return false;
        for (bh_intp i = 0; i < ndim; ++i)
            if (shape[i] != other.shape[i]) return false;
        for (bh_intp i = 0; i < ndim; ++i)
            if (stride[i] != other.stride[i]) return false;
        return true;
    }

    template<class Archive>
    void save(Archive & ar, const unsigned int version) const
    {
        size_t tmp = (size_t)base;
        ar << tmp;
        ar & start;
        ar & ndim;
        for(size_t i=0; i<ndim; ++i)
        {
            ar & shape[i];
            ar & stride[i];
        }
    }
    template<class Archive>
    void load(Archive & ar, const unsigned int version)
    {
        size_t tmp;
        ar >> tmp;
        base = (bh_base*)tmp;
        ar & start;
        ar & ndim;
        for(bh_intp i=0; i<ndim; ++i)
        {
            ar & shape[i];
            ar & stride[i];
        }
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()
};

/** Create a new base array.
 *
 * @param type The type of data in the array
 * @param nelements The number of elements
 * @param new_base The handler for the newly created base
 * @return Error code (BH_SUCCESS, BH_OUT_OF_MEMORY)
 */
DLLEXPORT bh_error bh_create_base(bh_type    type,
                                  bh_index   nelements,
                                  bh_base**  new_base);

/** Destroy the base array.
 *
 * @param base  The base array in question
 */
DLLEXPORT void bh_destroy_base(bh_base**  base);

DLLEXPORT std::ostream& operator<<(std::ostream& out, const bh_view& v);

#endif

