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

void compute()
{
    std::cout << "Hello Addition." << std::endl;

    BhArray<float> a({2,3,4}, {12,4,1});
    BhArray<float> b({2,3,4}, {12,4,1});
    BhArray<float> c({2,3,4}, {12,4,1});
    identity(b, 1);
    identity(c, 2);

    add(a, b, c);
    std::cout << a << std::endl;

    add(b, a, -10.0);
    std::cout << b << std::endl;

    Runtime::instance().flush();
}

int main()
{
    compute();
    return 0;
}

