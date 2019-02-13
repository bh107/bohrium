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
#include <iostream>

#include <bhxx/bhxx.hpp>

void compute() {
    using bhxx::BhArray;
    using bhxx::Runtime;

    BhArray<uint64_t> a({50, 3, 2});
    identity(a, 3);
    bhxx::random123(a, 42, 42);

    BhArray<uint64_t> b({3, 2});
    bhxx::add_reduce(b, a, 0);
    bhxx::free(a);

    BhArray<uint64_t> c({3, 2});
    bhxx::identity(c, b);

    BhArray<uint64_t> d({2});
    bhxx::add_reduce(d, c, 0);
    bhxx::free(c);

    std::cout << b << std::endl;
    std::cout << d << std::endl;
    Runtime::instance().flush();
}

int main() {
    compute();
    return 0;
}
