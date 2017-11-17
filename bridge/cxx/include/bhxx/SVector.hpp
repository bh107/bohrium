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

#include <bh_view.hpp>
#include <functional>
#include <numeric>
#include <ostream>
#include <vector>

namespace bhxx {

template <typename T, size_t MaxLength>
struct SVector : public std::vector<T> {
  public:
    using std::vector<T>::vector;
    SVector(const std::vector<T>& other) : SVector(other.begin(), other.end()) {}
    SVector() = default;

    T sum() const {
        return std::accumulate(this->begin(), this->end(), T{0});
    }

    T prod() const {
        return std::accumulate(this->begin(), this->end(), T{1}, std::multiplies<T>());
    }
};

// Some common SVectors
typedef SVector<int64_t, BH_MAXDIM> Stride;
typedef SVector<size_t, BH_MAXDIM>  Shape;

// Return a contiguous stride (row-major) based on `shape`
Stride contiguous_stride(const Shape& shape);

template <typename T, size_t MaxLength>
std::ostream& operator<<(std::ostream& o, const SVector<T, MaxLength>& vec) {
    o << '(';

    if (!vec.empty()) {
        auto it = std::begin(vec);
        o << *it;
        ++it;

        for (; it != std::end(vec); ++it) o << ',' << *it;
    }

    o << ')';
    return o;
}

}  // namespace bhxx
