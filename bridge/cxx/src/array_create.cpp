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
#include <bhxx/array_create.hpp>
#include <bhxx/random.hpp>

using namespace std;

namespace bhxx {

template<typename T>
BhArray<T> arange(int64_t start, int64_t stop, int64_t step) {
    if (step == 0) {
        throw std::overflow_error("Step cannot be zero");
    }

    // Let's make sure that 'step' is always positive
    bool swap_back = false;
    if (step < 0) {
        step *= -1;
        std::swap(start, stop);
        swap_back = true;
    }
    if (start >= stop) {
        throw std::overflow_error("Length of range cannot be zero");
    }
    auto size = static_cast<uint64_t>(std::ceil((static_cast<double>(stop) - static_cast<double>(start)) / step));

    // Get range 0..size
    BhArray<uint64_t> t1({size});
    bhxx::range(t1);
    // Cast to the correct return type
    BhArray<T> ret = bhxx::cast<T>(t1);

    // Handle the `start` and `stop` argument
    if (swap_back) {
        step *= -1;
        std::swap(start, stop);
    }
    if (step != 1) {
        bhxx::multiply(ret, ret, step);
    }
    if (start != 0) {
        bhxx::add(ret, ret, start);
    }
    return ret;
}


// Instantiate API that doesn't support booleans
#define INSTANTIATE(T)                         \
    template BhArray<T> arange<T>(int64_t start, int64_t stop, int64_t step);

instantiate_dtype()

#undef INSTANTIATE


}  // namespace bhxx
