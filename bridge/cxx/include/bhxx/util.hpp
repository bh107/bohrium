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

#include <sstream>
#include <algorithm>
#include <bhxx/BhArray.hpp>

namespace bhxx {

/** Force the execution of all lazy evaluated array operations */
void flush();

/** Create an contiguous view or a copy of an array.
 * The array is only copied if it isn't already contiguous.
 *
 * @tparam T   The data type of `ary`.
 * @param ary  The array to make contiguous.
 * @return     Either a view of `ary` or a new copy of `ary`.
 */
template<typename T>
BhArray<T> as_contiguous(BhArray<T> ary) {
    if (ary.isContiguous()) return std::move(ary);

    BhArray<T> contiguous{ary.shape()};
    identity(contiguous, ary);
    return contiguous;
}

/** Return the result of broadcasting `shapes` against each other
 *
 * @param shapes  Array of shapes
 * @return        Broadcasted shape
 */
template<int N>
Shape broadcasted_shape(std::array<Shape, N> shapes) {
    // Find the number of dimension of the broadcasted shape
    uint64_t ret_ndim = 0;
    for (const Shape &shape: shapes) {
        if (shape.size() > ret_ndim) {
            ret_ndim = shape.size();
        }
    }

    // Make sure that all shapes has the same length by prepending ones
    for (Shape &shape: shapes) {
        shape.insert(shape.begin(), ret_ndim - shape.size(), 1);
    }

    // The resulting shape is the max of each dimension
    Shape ret;
    for (uint64_t i = 0; i < ret_ndim; ++i) {
        uint64_t greatest = 0;
        for (const Shape &shape: shapes) {
            if (shape[i] > greatest) {
                greatest = shape[i];
            }
        }
        ret.push_back(greatest);
    }
    return ret;
}

/** Return a new view of `ary` that is broadcasted to `shape`
 *  We use the term broadcast as defined by NumPy. Let `ret` be the broadcasted view of `ary`:
 *    1) One-sized dimensions are prepended to `ret.shape()` until it has the same number of dimension as `ary`.
 *    2) The stride of each one-sized dimension in `ret` is set to zero.
 *    3) The shape of `ary` is set to `shape`
 *
 *  \note See: <https://docs.scipy.org/doc/numpy-1.15.0/user/basics.broadcasting.html>
 *
 * @param ary    Input array
 * @param shape  The new shape
 * @return       The broadcasted array
 */
template<typename T>
BhArray<T> broadcast_to(BhArray<T> ary, const Shape &shape) {
    if (ary.shape().size() > shape.size()) {
        std::stringstream ss;
        ss << "When broadcasting, the number of dimension of array (" << ary.shape().size()
           << ") cannot be greater than in the new shape (" << shape.size() << ")";
        throw std::runtime_error(ss.str());
    }
    // Prepend ones to shape and zeros to stride in order to make them the same lengths as `shape`
    Shape ret_shape = ary.shape();
    Stride ret_stride = ary.stride();
    assert(ret_shape.size() == ret_stride.size());
    ret_shape.insert(ret_shape.begin(), shape.size() - ret_shape.size(), 1);
    ret_stride.insert(ret_stride.begin(), shape.size() - ret_stride.size(), 0);

    // Broadcast each dimension by setting ret_stride to zero and ret_shape to `shape`
    for (uint64_t i = 0; i < shape.size(); ++i) {
        if (ret_shape[i] != shape[i]) {
            if (ret_shape[i] == 1) {
                ret_shape[i] = shape[i];
                ret_stride[i] = 0;
            } else {
                std::stringstream ss;
                ss << "Cannot broadcast shape " << ary.shape() << " to " << shape << ".";
                throw std::runtime_error(ss.str());
            }
        }
    }
    ary.setShapeAndStride(ret_shape, ret_stride);
    return std::move(ary);
}

/** Check whether `a` and `b` are the same view pointing to the same base
 *
 * @tparam T1 The data type of `a`.
 * @tparam T2 The data type of `b`.
 * @param a   The first array to compare.
 * @param b   The second array to compare.
 * @return    The boolean answer.
 */
template<typename T1, typename T2>
inline bool is_same_array(const BhArray<T1> &a, const BhArray<T2> &b) {
    if (a.base() == b.base() && a.offset() == b.offset() && a.shape() == b.shape()) {
        assert(a.shape().size() == b.shape().size());
        assert(a.stride().size() == b.stride().size());
        // Notice, the stride may vary when shape is one
        for (size_t i = 0; i < a.shape().size(); ++i) {
            if (a.shape()[i] > 1 && a.stride()[i] != b.stride()[i]) {
                return false;
            }
        }
        return true;
    } else {
        return false;
    }
}

/** Check whether `a` and `b` can share memory
 *
 *  @note A return of True does not necessarily mean that the two arrays share any element.
 *        It just means that they *might*.
 *
 * @tparam T1 The data type of `a`.
 * @tparam T2 The data type of `b`.
 * @param a   The first array to compare.
 * @param b   The second array to compare.
 * @return    The boolean answer.
 */
template<typename T1, typename T2>
bool may_share_memory(const BhArray<T1> &a, const BhArray<T2> &b) {
    assert(a.shape().size() == b.shape().size());
    assert(a.stride().size() == b.stride().size());

    if (a.base() != b.base()) {
        return false;
    }
    size_t a_low = a.offset();
    size_t b_low = b.offset();
    size_t a_high = a_low + 1;
    size_t b_high = b_low + 1;
    for (size_t i = 0; i < a.shape().size(); ++i) {
        if (a.stride()[i] < 0) {
            a_low += (a.shape()[i] - 1) * a.stride()[i];
        } else {
            a_high += (a.shape()[i] - 1) * a.stride()[i];
        }
        if (b.stride()[i] < 0) {
            b_low += (b.shape()[i] - 1) * b.stride()[i];
        } else {
            b_high += (b.shape()[i] - 1) * b.stride()[i];
        }
    }
    return !(b_low >= a_high || a_low >= b_high);
}

}  // namespace bhxx
