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
#include <vector>
#include <map>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/map.hpp>

// Forward declaration of class boost::serialization::access
namespace boost { namespace serialization { class access; }}


struct bh_slide_dim {
    /// The relevant dimension
    int64_t rank;

    /// Dimensions to be slided each loop iterations
    int64_t offset_change;

    /// The change to the shape
    int64_t shape_change;

    /// The strides these dimensions is slided each dynamically
    int64_t stride;

    /// The shape of the given dimension (used for negative indices)
    int64_t shape;

    /// The step delay in the dimension
    int64_t step_delay;

    template<class Archive>
    void serialize(Archive &ar, const unsigned int version) {
        ar & rank;
        ar & offset_change;
        ar & shape_change;
        ar & stride;
        ar & shape;
        ar & step_delay;
    }
};


struct bh_slide {
    bh_slide() = default;

    /// The slide in each dimension
    std::vector<bh_slide_dim> dims;

    /// Global iteration counter
    int64_t iteration_counter = 0;

    /// The amount the iterator can reach, before resetting it
    /// It maps a dimension to a pair of when to reset and a counter since last reset.
    std::map<int64_t, std::pair<int64_t, int64_t> > resets;

    /// Transposes by swapping the two axes 'axis1' and 'axis2'
    void transpose(int64_t axis1, int64_t axis2) {
        for (bh_slide_dim &dim: dims) {
            if (dim.rank == axis1) {
                dim.rank = axis2;
            } else if (dim.rank == axis2) {
                dim.rank = axis1;
            }
        }
        auto a1 = resets.find(axis1);
        auto a2 = resets.find(axis2);
        if (a1 != resets.end() and a2 != resets.end()) {
            auto tmp = a1->second;
            a1->second = a2->second;
            a2->second = tmp;
        } else if (a1 != resets.end()) {
            resets[axis2] = a1->second;
            resets.erase(a1);
        } else if (a2 != resets.end()) {
            resets[axis1] = a2->second;
            resets.erase(a2);
        }
    }

    template<class Archive>
    void serialize(Archive &ar, const unsigned int version) {
        ar & dims;
        ar & iteration_counter;
        ar & resets;
    }
};
