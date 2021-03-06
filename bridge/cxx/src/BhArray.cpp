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
#include <bhxx/array_create.hpp>

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

template<typename T>
const T *BhArray<T>::data(bool flush) const {
    if (_base == nullptr) {
        throw runtime_error("Array is uninitiated");
    }
    if (flush) {
        Runtime::instance().sync(_base);
        Runtime::instance().flush();
    }
    auto ret = static_cast<T *>(_base->getDataPtr());
    if (ret == nullptr) {
        return nullptr;
    } else {
        return _offset + ret;
    }
}

template<typename T>
bool BhArray<T>::isContiguous() const {
    assert(shape().size() == _stride.size());

    auto itshape = shape().rbegin();
    auto itstride = _stride.rbegin();

    int64_t acc = 1;
    for (; itstride != _stride.rend(); ++itstride, ++itshape) {
        if (*itshape > 1 && acc != *itstride) return false;
        acc *= static_cast<int64_t>(*itshape);
    }

    assert(acc == static_cast<int64_t>(size()));
    return offset() == 0;
}

template<typename T>
std::vector<T> BhArray<T>::vec() const {
    if (!isContiguous()) {
        throw runtime_error("Cannot call `vec()` on a non-contiguous array");
    }
    auto d = data();
    std::vector<T> ret(size());
    for (uint64_t i = 0; i < size(); ++i) {
        ret[i] = d[i];
    }
    return ret;
}

template<typename T>
void BhArray<T>::pprint(std::ostream &os, int current_nesting_level, int max_nesting_level) const {
    auto d = data();
    if (shape().empty()) {
        if (d == nullptr) {
            os << "null";
        } else {
            os << scientific;
            os << *d;
        }
    } else {
        os << "[";
        for (uint64_t i = 0; i < shape()[0]; ++i) {
            BhArray<T> t = (*this)[i];
            t.pprint(os, current_nesting_level+1, max_nesting_level);
            if (i < shape()[0] - 1) {
                os << ",";
                if (current_nesting_level < max_nesting_level) {
                    os << "\n";
                    for(int j=0; j<current_nesting_level + 1; ++j) {
                        os << " ";
                    }
                } else {
                    os << " ";
                }
            }
        }
        os << "]";
    }
}

template<typename T>
BhArray<T> BhArray<T>::operator[](int64_t idx) const {
    if (shape().empty()) {
        throw std::overflow_error("Cannot index a scalar, use `.data()` to access the scalar value");
    }
    // Negative index counts from the back
    if (idx < 0) {
        idx = shape()[0] + idx;
    }
    if (idx < 0 || idx >= static_cast<int64_t >(shape()[0])) {
        throw std::overflow_error("Index out of bound");
    }
    Shape ret_shape(shape().begin() + 1, shape().end());
    Stride ret_stride(_stride.begin() + 1, _stride.end());
    uint64_t ret_offset = offset() + idx * _stride[0];
    return BhArray<T>(base(), ret_shape, ret_stride, ret_offset);
}

template<typename T>
BhArray<T> BhArray<T>::transpose() const {
    Shape ret_shape(shape().rbegin(), shape().rend());
    Stride ret_stride(_stride.rbegin(), _stride.rend());
    return BhArray<T>{base(), std::move(ret_shape), std::move(ret_stride), offset()};
}

template<typename T>
BhArray<T> BhArray<T>::reshape(Shape shape) const {
    if (size() != shape.prod()) {
        throw std::runtime_error("Changing the shape cannot change the number of elements");
    }
    if (!isContiguous()) {
        throw std::runtime_error("Reshape not yet implemented for non-contiguous arrays.");
    }
    return BhArray<T>{base(), shape, contiguous_stride(shape), offset()};
}

template<typename T>
BhArray<T> BhArray<T>::newAxis(int axis) const {
    const auto ndim = static_cast<int>(shape().size());
    // Negative index counts from the back
    if (axis < 0) {
        axis = ndim + axis + 1;
    }
    if (axis < 0 || axis > ndim) {
        throw std::overflow_error("Axis out of bound");
    }
    Shape ret_shape(shape());
    Stride ret_stride(stride());
    ret_shape.insert(ret_shape.begin() + axis, 1, 1);
    ret_stride.insert(ret_stride.begin() + axis, 1, 0);
    BhArray<T> ret{*this};
    ret.setShapeAndStride(std::move(ret_shape), std::move(ret_stride));
    return ret;
}

#define INSTANTIATE(TYPE) template class BhArray<TYPE>;

instantiate_dtype()

#undef INSTANTIATE

}  // namespace bhxx
