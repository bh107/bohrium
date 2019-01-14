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

using namespace bhxx;

void compute() {
    std::cout << "Hello Indexing." << std::endl;

    BhArray<double> a;
    {
        BhArray<uint64_t> t1({2 * 3});
        range(t1);
        auto t2 = decltype(a)({2 * 3});
        identity(t2, t1);
        auto t3 = reshape(t2, {2, 3});
        a = t3;
    }
    std::cout << a << std::endl;
    auto b = a[1][1];
    std::cout << b << std::endl;
}

int main() {
    compute();
    return 0;
}

