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

#include <bhxx/random.hpp>
#include <bhxx/type_traits_util.hpp>

using namespace std;

namespace bhxx {

// Default instance of the random number generation
Random random;

template<typename T>
BhArray<T> Random::randn(Shape shape) {
    BhArray<T> ary(random.random123(shape.prod()));
    T max_value = static_cast<T>(std::numeric_limits<uint64_t>::max());
    return (ary / max_value).reshape(shape);
}

// Instantiate API that doesn't support booleans
#define INSTANTIATE(T)                         \
    template BhArray<T> Random::randn(Shape shape);

instantiate_dtype_float()

#undef INSTANTIATE

} // namespace bhxx
