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
#include "bh_cppb.hpp"

int main() {

    bh::init();
    bh::Vector<float> a = bh::Vector<float>(1024, 1024);
    bh::Vector<float> b = bh::Vector<float>(1024, 1024);
    bh::Vector<float> c = bh::Vector<float>(1024, 1024);

    //
    // Beautiful code
    //
    a = (float)1;
    b = (float)2;
    a = (float)3;
    c = a;
    ++c;
    c++;
    --c;
    c--;
    c = ((a+(float)1)++)--;

    c = (float)1+a;
    c = a+b;

    std::cout << "And the value is: " << a << "." << std::endl;
    std::cout << "Flushing " << bh::flush() << " operations" << std::endl;

    bh::shutdown();

    return 0;

}
