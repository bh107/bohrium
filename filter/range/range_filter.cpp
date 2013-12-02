/*
This file is part of Bohrium and copyright (c) 2012 the Bohrium
team <http://www.bh107.org>.

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
#include <sstream>
#include <iostream>
#include <string>
#include <vector>
#include <set>
#include <bh.h>

using namespace std;

void range_filter(bh_ir* bhir)
{
    bh_intp nodes[] = {0};
    cout << "### Range-filter searching through. ndag="<< bhir->ndag  << "." << endl;
    bh_dag_split(bhir, 1, nodes, 0, -1);  
    cout << "###" << endl;
}

