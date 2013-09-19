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
#include "../../../cpp/bh/bh.hpp"

using namespace std;
using namespace bh;

void compute()
{
    cout << "Hello World." << endl;

	int i;

	multi_array<double> x, v, y;
	
	//for (i = 50; i < 1000; i++) {
	i = 1000;

	x = zeros<double>(i, i);
	v = x[_(1, 49, 1)][_(1, 49, 1)];
	y = v * 0.2;

	Runtime::instance().flush();
	//}

    //cout << x * 0.2 << endl;

    std::cout << "Leaving compute!" << std::endl;
}

int main()
{
    compute();
    return 0;
}

