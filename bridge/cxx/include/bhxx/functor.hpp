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

#include <bhxx/array_operations.hpp>

namespace bhxx {

struct OpBase {
    Shape result_shape(const Shape& lhsshape, const Shape& rhsshape) {
        if (lhsshape != rhsshape) {
            throw std::runtime_error("Shapes of lhs and rhs do not agree.");
        }
        return lhsshape;
    }

    Shape reduction_shape(Shape fullshape, int64_t axis) {
        if (axis < 0 || static_cast<size_t>(axis) >= fullshape.size()) {
            throw std::runtime_error(
                  "Axis needs to be larger or equal to zero and smaller than the rank.");
        }

        if (fullshape.size() > 1) {
            fullshape.erase(fullshape.begin() + axis);
            return fullshape;
        } else if (fullshape.size() == 1) {
            return {1};  // Full reduction
        } else {
            throw std::runtime_error("Cannot reduce an array of shape 0");
            return {1};
        }
    }
};

template <typename T>
struct Add : public OpBase {
    BhArray<T> operator()(const BhArray<T>& lhs, const BhArray<T>& rhs) {
        BhArray<T> result(result_shape(lhs.shape(), rhs.shape()));
        add(result, lhs, rhs);
        return result;
    }
};

template <typename T>
struct Multiply : public OpBase {
    BhArray<T> operator()(const BhArray<T>& lhs, const BhArray<T>& rhs) {
        BhArray<T> result(result_shape(lhs.shape(), rhs.shape()));
        multiply(result, lhs, rhs);
        return result;
    }
};

template <typename T>
struct Equal : public OpBase {
    BhArray<bool> operator()(const BhArray<T>& lhs, const BhArray<T>& rhs) {
        BhArray<bool> result(result_shape(lhs.shape(), rhs.shape()));
        equal(result, lhs, rhs);
        return result;
    }
};

template <typename T>
struct NotEqual : public OpBase {
    BhArray<bool> operator()(const BhArray<T>& lhs, const BhArray<T>& rhs) {
        BhArray<bool> result(result_shape(lhs.shape(), rhs.shape()));
        not_equal(result, lhs, rhs);
        return result;
    }
};

template <typename T>
struct AddReduce : public OpBase {
    BhArray<T> operator()(const BhArray<T>& lhs, int64_t axis) {
        BhArray<T> result(reduction_shape(lhs.shape(), axis));
        add_reduce(result, lhs, axis);
        return result;
    }
};

template <typename T>
struct LogicalAndReduce : public OpBase {
    BhArray<T> operator()(const BhArray<T>& lhs, int64_t axis) {
        BhArray<T> result(reduction_shape(lhs.shape(), axis));
        logical_and_reduce(result, lhs, axis);
        return result;
    }
};

template <typename T>
struct LogicalOrReduce : public OpBase {
    BhArray<T> operator()(const BhArray<T>& lhs, int64_t axis) {
        BhArray<T> result(reduction_shape(lhs.shape(), axis));
        logical_or_reduce(result, lhs, axis);
        return result;
    }
};

}  // namespace bhxx
