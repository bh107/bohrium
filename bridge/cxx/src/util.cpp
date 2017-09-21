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

#include <bhxx/Runtime.hpp>
#include <bhxx/array_operations.hpp>
#include <bhxx/util.hpp>
#include <complex>

namespace bhxx {

template <typename T>
T as_scalar(BhArray<T> ary) {
    if (ary.base == nullptr) {
        throw std::runtime_error(
              "Cannot call bhxx::as_scalar on BhArray objects without base");
    }

    if (ary.n_elem() != 1) {
        throw std::runtime_error(
              "Cannot call bhxx::as_scalar on BhArray objects with more than one "
              "element");
    }

    Runtime::instance().sync(ary.base);
    Runtime::instance().flush();

    const T* data = ary.data();
    if (data == nullptr) {
        throw std::runtime_error("Cannot get the scalar from an uninitialised BhArray.");
    }

    return *data;
}

template <typename T>
BhArray<T> broadcast(BhArray<T> ary, int64_t axis, size_t size) {
    if (axis < 0 || static_cast<size_t>(axis) > ary.rank()) {
        throw std::runtime_error(
              "Axis to replicate needs to be larger than -1 and less than or equal to "
              "the rank of the array.");
    }
    if (size == 0) throw std::runtime_error("The new size needs to be larger than 0");
    auto& shape  = ary.shape;
    auto& stride = ary.stride;

    shape.insert(shape.begin() + axis, size);
    stride.insert(stride.begin() + axis, 0);
    return ary;
}

template <typename T>
BhArray<T> transpose(BhArray<T> ary) {
    std::reverse(ary.shape.begin(), ary.shape.end());
    std::reverse(ary.stride.begin(), ary.stride.end());
    return ary;
}

template <typename T>
BhArray<T> reshape(BhArray<T> ary, Shape shape) {
    if (ary.n_elem() != shape.prod()) {
        throw std::runtime_error(
              "Changing the shape cannot change the number of elements");
    }

    if (ary.shape == shape) return ary;

    if (!ary.is_contiguous()) {
        throw std::runtime_error(
              "Reshape not yet implemented for non-contiguous arrays.");
    }

    ary.shape  = shape;
    ary.stride = contiguous_stride(shape);
    return ary;
}

template <typename T>
BhArray<T> matmul(BhArray<T> lhs, BhArray<T> rhs) {
    if (lhs.rank() == 0 || rhs.rank() == 0) {
        throw std::runtime_error("Lhs and Rhs need to be of at least rank 1.");
    }

    // Check that the axis we contract over is common.
    if (lhs.shape.back() != rhs.shape.front()) {
        throw std::runtime_error("Common axis of arrays has incompatible sizes. LHS == " +
                                 std::to_string(lhs.shape.back()) + ", RHS == " +
                                 std::to_string(rhs.shape.front()) + ".");
    }

    if (lhs.rank() > 2 || rhs.rank() > 2) {
        throw std::runtime_error("matmul not implemented for arrays with rank > 2.");
    }

    // Make lhs and rhs matrices in case they are not
    // (i.e. prepend/append 1 to the dimension if they are vectors)
    // and build the resulting shapes
    //
    // Notice that the case where both and lhs and rhs have rank 1
    // is also covered, since then both branches will be executed
    // leading to a result_shape of {1}.
    Shape result_shape{lhs.shape.front(), rhs.shape.back()};
    if (lhs.rank() == 1) {
        result_shape = {rhs.shape.back()};
        lhs          = reshape(std::move(lhs), {1, lhs.n_elem()});
    }
    if (rhs.rank() == 1) {
        result_shape = {lhs.shape.front()};
        rhs          = reshape(std::move(rhs), {rhs.n_elem(), 1});
    }

    BhArray<T> result({lhs.shape.front(), rhs.shape.back()});
    try {
        lhs = as_contiguous(std::move(lhs));
        rhs = as_contiguous(std::move(rhs));
        Runtime::instance().enqueue_extmethod("blas_gemm", result, lhs, rhs);
    } catch (...) {
        // No blas could be found.
        // TODO Use check function once it is available

        // Broadcast lhs and a transposed rhs
        Shape      broad_shape{lhs.shape.front(), rhs.shape.back(), rhs.shape.front()};
        BhArray<T> rhs_trans = transpose(std::move(rhs));
        BhArray<T> lhs_broad = broadcast(std::move(lhs), 1, broad_shape[1]);
        BhArray<T> rhs_broad = broadcast(std::move(rhs_trans), 0, broad_shape[0]);
        assert(lhs_broad.shape == rhs_broad.shape);
        assert(lhs_broad.shape == broad_shape);

        // Multiply and reduce
        BhArray<T> tmp(broad_shape);
        multiply(tmp, lhs_broad, rhs_broad);
        add_reduce(result, tmp, 2);
    }

    return reshape(std::move(result), result_shape);
}

// Instantiate all possible types of `BhArray`
#define INSTANTIATE(T)                         \
    template T          as_scalar(BhArray<T>); \
    template BhArray<T> transpose(BhArray<T>); \
    template BhArray<T> broadcast(BhArray<T>, int64_t, size_t)

#define INSTANTIATE_NOBOOL(T) \
    INSTANTIATE(T);           \
    template BhArray<T> matmul(BhArray<T>, BhArray<T>)

INSTANTIATE(bool);
INSTANTIATE_NOBOOL(int8_t);
INSTANTIATE_NOBOOL(int16_t);
INSTANTIATE_NOBOOL(int32_t);
INSTANTIATE_NOBOOL(int64_t);
INSTANTIATE_NOBOOL(uint8_t);
INSTANTIATE_NOBOOL(uint16_t);
INSTANTIATE_NOBOOL(uint32_t);
INSTANTIATE_NOBOOL(uint64_t);
INSTANTIATE_NOBOOL(float);
INSTANTIATE_NOBOOL(double);
INSTANTIATE_NOBOOL(std::complex<float>);
INSTANTIATE_NOBOOL(std::complex<double>);

#undef INSTANTIATE
#undef INSTANTIATE_NOBOOL

}  // namespace bhxx
