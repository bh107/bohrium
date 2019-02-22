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

#include <cstdint>
#include <bhxx/BhArray.hpp>
#include <bhxx/array_operations.hpp>


namespace bhxx {

/** Return a new empty array
 *
 * @tparam T    The data type of the new array
 * @param shape The shape of the new array
 * @return      The new array
 */
template<typename T>
BhArray <T> empty(Shape shape) {
    return BhArray<T>{std::move(shape)};
}

/** Return a new empty array that has the same shape as `ary`
 *
 * @tparam OutType  The data type of the returned new array
 * @tparam InType   The data type of the input array
 * @param ary       The array to take the shape from
 * @return          The new array
 */
template<typename OutType, typename InType>
BhArray <OutType> empty_like(const bhxx::BhArray<InType> &ary) {
    return BhArray<OutType>{ary.shape()};
}

/** Return a new array filled with `value`
 *
 * @tparam T    The data type of the new array
 * @param shape The shape of the new array
 * @param value The value to fill the new array with
 * @return      The new array
 */
template<typename T>
BhArray <T> full(Shape shape, T value) {
    BhArray<T> ret{std::move(shape)};
    ret = value;
    return ret;
}

/** Return a new array filled with zeros
 *
 * @tparam T    The data type of the new array
 * @param shape The shape of the new array
 * @return      The new array
 */
template<typename T>
BhArray <T> zeros(Shape shape) {
    return full(std::move(shape), T{0});
}

/** Return a new array filled with ones
 *
 * @tparam T    The data type of the new array
 * @param shape The shape of the new array
 * @return      The new array
 */
template<typename T>
BhArray <T> ones(Shape shape) {
    return full(std::move(shape), T{1});
}

/** Return evenly spaced values within a given interval.
 *
 * @tparam T     Data type of the returned array
 * @param start  Start of interval. The interval includes this value.
 * @param stop   End of interval. The interval does not include this value.
 * @param step   Spacing between values. For any output out, this is the distance between
 *               two adjacent values, out[i+1] - out[i].
 * @return       New 1D array
 */
template<typename T>
BhArray <T> arange(int64_t start, int64_t stop, int64_t step);

/** Return evenly spaced values within a given interval using steps of 1.
 *
 * @tparam T     Data type of the returned array
 * @param start  Start of interval. The interval includes this value.
 * @param stop   End of interval. The interval does not include this value.
 * @return       New 1D array
 */
template<typename T>
BhArray <T> arange(int64_t start, int64_t stop) {
    return arange<T>(start, stop, 1);
}

/** Return evenly spaced values from 0 to `stop` using steps of 1.
 *
 * @tparam T     Data type of the returned array
 * @param stop   End of interval. The interval does not include this value.
 * @return       New 1D array
 */
template<typename T>
BhArray <T> arange(int64_t stop) {
    return arange<T>(0, stop, 1);
}

/** Element-wise `static_cast`.
 *
 * @tparam OutType  The data type of the returned array
 * @tparam InType   The data type of the input array
 * @param ary       Input array to cast
 * @return          New array
 */
template<typename OutType, typename InType>
BhArray <OutType> cast(const bhxx::BhArray<InType> &ary) {
    BhArray<OutType> ret = empty_like<OutType>(ary);
    bhxx::identity(ret, ary);
    return ret;
}

}  // namespace bhxx
