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
#include "bh/bh.hpp"

using namespace std;
using namespace bh;

void compute()
{
    /*
    cout << "Assign to slice" << endl;

    multi_array<float> x(9,9);
    x = (float)1.0;
    Runtime::instance().flush();

    std::cout << "\n\n first row" << std::endl;
    x[0][_(0,9,1)] = 1.0;
    Runtime::instance().flush();

    std::cout << "\n\n last row" << std::endl;
    x[-1][_(0,9,1)] = 1.0;
    Runtime::instance().flush();

    std::cout << "\n\n inbetween row" << std::endl;
    x[4][_(0,9,1)] = 1.0;
    Runtime::instance().flush();
    */

    /*
    cout << "Assign to slice2" << endl;

    multi_array<float> x(9,9);
    x = (float)1.0;
    Runtime::instance().flush();

    std::cout << "\n\n first row" << std::endl;
    x[_(0,9,1)][0] = 1.0;
    Runtime::instance().flush();

    std::cout << "\n\n last row" << std::endl;
    x[_(0,9,1)][-1] = 1.0;
    Runtime::instance().flush();

    std::cout << "\n\n inbetween row" << std::endl;
    x[_(0,9,1)][4] = 1.0;
    Runtime::instance().flush();

    */

    /*
    cout << "Assign to slice single-element" << endl;

    multi_array<float> x(9,9);
    x = (float)1.0;
    Runtime::instance().flush();
    cout << x << endl;

    std::cout << "\n\n first & first" << std::endl;
    x[0][0] = 3.0;
    Runtime::instance().flush();
    cout << x << endl;

    std::cout << "\n\n in-between" << std::endl;
    x[4][4] = 3.0;
    Runtime::instance().flush();
    cout << x << endl;

    std::cout << "\n\n last & last" << std::endl;
    x[-1][-1] = 3.0;
    Runtime::instance().flush();
    cout << x << endl;
    */

    cout << "Assign to slice of tensor" << endl;
    multi_array<float> x(9,9,9);
    x = (float)1.0;
    Runtime::instance().flush();
    cout << x << endl;

    std::cout << "\n\n (0,0,0)" << std::endl;
    x[0][0][0] = 3.0;
    Runtime::instance().flush();
    cout << x << endl;

    std::cout << "\n\n (4,4,4)" << std::endl;
    x[4][4][4] = 3.0;
    Runtime::instance().flush();
    cout << x << endl;

    std::cout << "\n\n (8,8,8)" << std::endl;
    x[-1][-1][-1] = 3.0;
    Runtime::instance().flush();
    cout << x << endl;

}

int main()
{
    std::cout << "COMPUTE!" << std::endl;
    compute();
    std::cout << "STOPPED!" << std::endl;
    return 0;
}

