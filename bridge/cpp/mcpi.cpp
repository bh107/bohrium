/*
This file is part of cphVB and copyright (c) 2012 the cphVB team:
http://cphvb.bitbucket.org

cphVB is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as 
published by the Free Software Foundation, either version 3 
of the License, or (at your option) any later version.

cphVB is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the 
GNU Lesser General Public License along with cphVB. 

If not, see <http://www.gnu.org/licenses/>.
*/
#include <iostream>
#include "cphvb_cppb.hpp"

int main() {

    cphvb::init();
    cphvb::Vector<float> a = cphvb::Vector<float>(1024, 1024);
    cphvb::Vector<float> b = cphvb::Vector<float>(1024, 1024);
    cphvb::Vector<float> c = cphvb::Vector<float>(1024, 1024);

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
    std::cout << "Flushing " << cphvb::flush() << " operations" << std::endl;

    cphvb::shutdown();

    return 0;

}
