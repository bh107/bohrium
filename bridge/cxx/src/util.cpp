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
T as_scalar(BhArray<T>& ary) {
    if (ary.base == nullptr) {
        throw std::runtime_error(
              "Cannot call bhxx::as_scalar on BhArray objects without base");
    }

    if (ary.n_elem() != 1) {
        throw std::runtime_error(
              "Cannot call bhxx::as_scalar on BhArray objects with more than one "
              "element");
    }

    sync(ary);
    Runtime::instance().flush();

    const T* data = ary.data();
    if (data == nullptr) {
        throw std::runtime_error("Cannot get the scalar from an uninitialised BhArray.");
    }

    return *data;
}


// Instantiate all possible types of `BhArray`
#define INSTANTIATE(T)                          \
    template T          as_scalar(BhArray<T>&); \

INSTANTIATE(bool);
INSTANTIATE(int8_t);
INSTANTIATE(int16_t);
INSTANTIATE(int32_t);
INSTANTIATE(int64_t);
INSTANTIATE(uint8_t);
INSTANTIATE(uint16_t);
INSTANTIATE(uint32_t);
INSTANTIATE(uint64_t);
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(std::complex<float>);
INSTANTIATE(std::complex<double>);

#undef INSTANTIATE

}  // namespace bhxx
