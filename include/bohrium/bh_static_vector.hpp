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

/// The maximum number of possible dimension in arrays.
constexpr int64_t BH_MAXDIM = 16;


/** We use a static allocated vector with a maximum capacity of `BH_MAXDIM`
    Notice, `BhStaticVector<T>` and `std::vector<T>` is interchangeable as long
    as the vector is smaller than `BH_MAXDIM`.
*/
template<typename T>
class BhStaticVector : public boost::container::static_vector<T, BH_MAXDIM> {
public:

    // This is C++11 syntax for exposing the constructor from the parent class
    using boost::container::static_vector<T, BH_MAXDIM>::static_vector;

    /// The sum of the elements in this vector
    virtual T sum() const {
        return std::accumulate(this->begin(), this->end(), T{0});
    }

    /// The product of the elements in this vector
    virtual T prod() const {
        return std::accumulate(this->begin(), this->end(), T{1}, std::multiplies<T>());
    }

    /// Pretty printing of this vector
    virtual std::string pprint() const {
        std::stringstream ss;
        ss << '(';
        if (!this->empty()) {
            auto it = this->begin();
            ss << *it;
            ++it;
            for (; it != this->end(); ++it) ss << ',' << *it;
        }
        ss << ')';
        return ss.str();
    }
};

/// Print overload
template<typename T>
inline std::ostream &operator<<(std::ostream &o, const BhStaticVector<T> &vec) {
    o << vec.pprint();
    return o;
}

/// The type used throughout Bohrium
typedef BhStaticVector<int64_t> BhIntVec;