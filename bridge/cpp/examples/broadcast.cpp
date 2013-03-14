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
    multi_array<double> x(9,3);
    multi_array<double> s(3);
    multi_array<double> t(9,3);

    x = 3.0;

    s = 6.0;
    t = 8.0;

    std::cout << "[Broadcast: Matrix = Vector]" << std::endl;
    x = s;
    pprint(x);

    try {
        std::cout << "[Broadcast: Vector = Matrix]" << std::endl;
        s = x;
        pprint(x);
    } catch (std::exception& e) {
        std::cout << e.what() << std::endl;
    }

    std::cout << "[Broadcast: Vector + Matrix]" << std::endl;
    x = s + t;
    pprint(x);

    std::cout << "[Broadcast: Matrix + Vector]" << std::endl;
    x = t + s;
    pprint(x);

    std::cout << "[Broadcast manual]" << std::endl;
    multi_array<double> vector(3);
    multi_array<double> matrix(9,3);
    multi_array<double> tensor(3,9,3);
    multi_array<double>& view = Runtime::instance()->view(vector);

    vector = 2.0;
    matrix = 3.0;
    tensor = 3.0;

    std::cout << "<DOIT>" << std::endl;
    //broadcast(vector, matrix, view);
    broadcast(vector, tensor, view);
    pprint( tensor + view );

}

int main()
{
    std::cout << "Broadcast example." << std::endl;

    compute();

    return 0;
}

