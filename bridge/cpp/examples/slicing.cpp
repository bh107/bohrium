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

using namespace std;
using namespace bh;

int main()
{
    cout << "Slicing." << endl;

    multi_array<double> x(9,9);
    multi_array<double>& y = Runtime::instance()->view(x);
    x = 1.0;
    bh_pprint_array(&storage[y.getKey()]);
    //y = x[ALL][_(0,8,2)];
    bh_pprint_array(&storage[y.getKey()]);

    //bh_pprint_array(y);
    cout << y;

    return 0;
}

