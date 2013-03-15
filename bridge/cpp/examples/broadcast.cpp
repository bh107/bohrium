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

void m_v()
{
    std::cout << "\n\n{[ Broadcast: Matrix = Vector" << std::endl;
    multi_array<double> m(9,3);
    multi_array<double> v(3);

    m = 2.0;
    v = 3.0;

    std::cout << "<<NOW>>" << std::endl;
    m = v;
    pprint(m);
    std::cout << "]}" << std::endl;
}

void v_m()
{
    std::cout << "\n\n{[ Broadcast: Vector = Matrix" << std::endl;
    multi_array<double> m(9,3);
    multi_array<double> v(3);

    m = 2.0;
    v = 3.0;

    std::cout << "<<NOW>>" << std::endl;
    try {
        v = m;
    } catch (std::exception& e) {
        std::cout << e.what() << std::endl;
    }
    pprint(v);
    std::cout << "]}" << std::endl;
}

void mv()
{
    std::cout << "\n\n{[ Broadcast: Matrix + Vector" << std::endl;
    multi_array<double> m(9,3);
    multi_array<double> v(3);

    m = 2.0;
    v = 3.0;

    std::cout << "<<NOW>>" << std::endl;
    m+v;
    //pprint(m+v);
    std::cout << "]}" << std::endl;
}

void vm()
{
    std::cout << "\n\n{[ Broadcast: Vector + Matrix" << std::endl;
    multi_array<double> m(9,3);
    multi_array<double> v(3);

    m = 2.0;
    v = 3.0;

    std::cout << "<<NOW>>" << std::endl;
    pprint(v+m);
    std::cout << "]}" << std::endl;
}

int main()
{
    std::cout << "<< Broadcast example >>" << std::endl;
    m_v();
    vm();
    mv();

    v_m();

    return 0;
}

