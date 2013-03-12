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
#include <iostream>
#include "bh/cppb.hpp"

using namespace bh;

void compute()
{
    multi_array<double> x(3);
    multi_array<double> y(9,3);
    multi_array<double> z(9,3);

    x = 2.0;
    y = 3.0;

    std::cout << "Compatible? " << broadcast_shape(x, y, z) << "." << std::endl;
    z = x + y;
    pprint(z);
}

int main()
{
    std::cout << "Broadcast example." << std::endl;

    compute();

    return 0;
}

