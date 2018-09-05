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
    int64_t dim;

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
        ar & dim;
        ar & offset_change;
        ar & shape_change;
        ar & stride;
        ar & shape;
        ar & step_delay;
    }
};


struct bh_slide {
    bh_slide() = default;

    std::vector<bh_slide_dim> dims;

    int64_t iteration_counter = 0;

    // The amount the iterator can reach, before resetting it
    std::map<int64_t, int64_t> resets;
    std::map<int64_t, int64_t> changes_since_reset;

    template<class Archive>
    void serialize(Archive &ar, const unsigned int version) {
        ar & dims;
        ar & iteration_counter;
        ar & resets;
        ar & changes_since_reset;
    }
};
