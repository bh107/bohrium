//
// Copyright (C) 2017 by the linalgwrap authors
//
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

#include <functional>
#include <numeric>
#include <ostream>
#include <boost/container/static_vector.hpp>
#include <sstream>

// The maximum number of possible dimension in arrays.
constexpr int64_t BH_MAXDIM = 16;


/** We use a static allocated vector with a maximum capacity of `BH_MAXDIM`
    Notice, `BhIntVec` and `std::vector<int64_t>` is interchangeable as long
    as the vector is smaller than `BH_MAXDIM`.
*/
class BhIntVec : public boost::container::static_vector<int64_t, BH_MAXDIM> {
public:

    // This is C++11 syntax for exposing the constructor from the parent class
    using boost::container::static_vector<int64_t, BH_MAXDIM>::static_vector;

    /// The sum of the elements in this vector
    int64_t sum() const {
        return std::accumulate(this->begin(), this->end(), int64_t{0});
    }

    /// The product of the elements in this vector
    int64_t prod() const {
        return std::accumulate(this->begin(), this->end(), int64_t{1}, std::multiplies<int64_t>());
    }

    /// Pretty printing of this vector
    std::string pprint() const {
        std::stringstream ss;
        ss << '(';
        if (!empty()) {
            auto it = begin();
            ss << *it;
            ++it;
            for (; it != end(); ++it) ss << ',' << *it;
        }
        ss << ')';
        return ss.str();
    }
};

/// Print overload
inline std::ostream &operator<<(std::ostream &o, const BhIntVec &vec) {
    o << vec.pprint();
    return o;
}
