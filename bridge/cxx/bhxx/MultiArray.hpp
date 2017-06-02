//
// Copyright (C) 2017 by the linalgwrap authors
//
// This file is part of linalgwrap.
//
// linalgwrap is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published
// by the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// linalgwrap is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with linalgwrap. If not, see <http://www.gnu.org/licenses/>.
//

#include <vector>
#include <numeric>
#include <bh_component.hpp>
#include <bxx/traits.hpp>


namespace bhxx {

template <typename T, std::size_t MaxLength>
struct SVector : public std::vector<T> {
public:
    using std::vector<T>::vector;
    SVector(const std::vector<T> &other) : SVector(other.begin(), other.end()) {}
    SVector() = default;

    T sum() const {
        return std::accumulate(this->begin(), this->end(), T{0});
    }
    T prod() const {
        return std::accumulate(this->begin(), this->end(), T{1}, std::multiplies<T>());
    }
};


template <typename T>
class BhBase {
    bh_base base;
public:
    typedef T scalar_type;
    BhBase(size_t nelem) {
        base.data = nullptr;
        base.nelem = nelem;
        bxx::assign_array_type<T>(&base);
    }
};


template <typename T>
class BhArray {
public:
    typedef T scalar_type;
    SVector<size_t, BH_MAXDIM> shape;
    std::shared_ptr<BhBase<T> > base;

    // Create a new view that points to a new base
    BhArray(const std::vector<size_t> &new_shape) : shape(new_shape), base(new BhBase<T>(shape.prod())) {}

    // Create a new view that points to base that `other` points to
    BhArray(const std::vector<size_t> &new_shape, BhArray other) : shape(new_shape), base(std::move(other.base)) {}
};



}