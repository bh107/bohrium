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
    Vector<double> x = Vector<double>(3,3);
    Vector<double> y = Vector<double>(3,3);
    Vector<double> z = Vector<double>(3,3);

    y = 1.0;
    x = 1.0;
    
    x += 1.0;
    pprint( x );

    x += y;
    pprint( x );
}

int main()
{
    std::cout << "Compound example." << std::endl;

    compute();

    return 0;
}

