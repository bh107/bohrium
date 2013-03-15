/*
This file is part of Bohrium and copyright (c) 2012 the Bohrium team:
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
GNU Lesser General Public License along with Bohrium. 

If not, see <http://www.gnu.org/licenses/>.
*/
#include <iostream>

namespace bh {

template <typename T>
vector<T>::vector(unsigned int n) : multi_array<T>(n) {
    DEBUG_PRINT("<Vector>\n");
};

/*

This might be fun to do an initializer on the form:

x = 1,2,3,
    4,5,6,
    7,8,8;

*/
template <typename T>
vector<T>& operator, ( vector<T>& lhs, T rhs )
{
    DEBUG_PRINT("[%d,%d]", lhs.getKey(), rhs);
    return lhs;
}


}

