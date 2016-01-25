/*
This file is part of Bohrium and Copyright (c) 2012 the Bohrium team:
http://bohrium.bitbucket.org

Bohrium is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3
of the License, or (at your option) any later version.

Bohrium is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the
GNU Lesser General Public License along with bohrium.

If not, see <http://www.gnu.org/licenses/>.
*/
#include <bxx/bohrium.hpp>

#include <iostream>
#include <complex>

using namespace bxx;

void compute()
{
    std::cout << "Hello Addition." << std::endl;

    multi_array<float> x, y;
    
    x = ones<float>(10);
    y = ones<float>(10);

    std::cout << x + y << std::endl;
}

int main()
{
    compute();
    return 0;
}

