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

#include <bhxx/BhArray.hpp>
#include <bhxx/Runtime.hpp>
#include <bhxx/array_operations.hpp>
#include <bhxx/util.hpp>

using namespace std;

namespace bhxx {

// Note: This one line of code cannot move to the hpp file,
// since it requires the inclusion of Runtime.hpp, which in turn
// requires the inclusion of BhArray.hpp
void RuntimeDeleter::operator()(BhBase *ptr) const {
    // Simply hand the deletion over to Bohrium
    // including the ownership of the pointer to be deleted
    // by the means of a unique pointer.
    Runtime::instance().enqueueDeletion(std::unique_ptr<BhBase>(ptr));
}

//
// Properties and data access
//

template<typename T>
bool BhArray<T>::isContiguous() const {
    assert(shape.size() == stride.size());

    auto itshape = shape.rbegin();
    auto itstride = stride.rbegin();

    int64_t acc = 1;
    for (; itstride != stride.rend(); ++itstride, ++itshape) {
        if (*itstride > 1 && acc != *itstride) return false;
        acc *= static_cast<int64_t>(*itshape);
    }

    assert(acc == static_cast<int64_t>(size()));
    return offset == 0;
}

//
// Routines
//

template<typename T>
void BhArray<T>::pprint(std::ostream &os) const {
    if (base == nullptr) {
        throw runtime_error("Cannot call pprint on array without base");
    }
    Runtime::instance().sync(base);
    Runtime::instance().flush();

    if (shape.empty()) {
        if (data() == nullptr) {
            os << "null";
        } else {
            os << scientific;
            os << *data();
        }
    } else {
        os << "[";
        for (uint64_t i = 0; i < shape[0]; ++i) {
            BhArray<T> t = (*this)[i];
            t.pprint(os);
            if (i < shape[0] - 1) {
                os << ",";
            }
        }
        os << "]";
    }
}

template<typename T>
BhArray<T> BhArray<T>::operator[](int64_t idx) const {
    if (shape.empty()) {
        throw std::overflow_error("Cannot index a scalar, use `.data()` to access the scalar value");
    }
    // Negative index counts from the back
    if (idx < 0) {
        idx = shape[0] + idx;
    }
    if (idx < 0 || idx >= static_cast<int64_t >(shape[0])) {
        throw std::overflow_error("Index out of bound");
    }
    Shape ret_shape(shape.begin() + 1, shape.end());
    Stride ret_stride(stride.begin() + 1, stride.end());
    uint64_t ret_offset = offset + idx * stride[0];
    return BhArray<T>(base, ret_shape, ret_stride, ret_offset);
}


#define INSTANTIATE(TYPE) template class BhArray<TYPE>;

instantiate_dtype()

#undef INSTANTIATE

}  // namespace bhxx
