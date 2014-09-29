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
#include <bh.h>
#include <iostream>
#include <fstream>
#include <boost/foreach.hpp>
#include <vector>
#include <map>
#include <stdexcept>

using namespace std;
using namespace boost;

void fuser(bh_ir &bhir)
{
    if(bhir.kernel_list.size() != 0)
    {
        throw logic_error("The kernel_list is not empty!");
    }
    //map from an instruction in the kernel_list to an index into the
    //original instruction list
    map<pair<int,int>,int> index_map;

    vector<bh_instruction>::iterator it = bhir.instr_list.begin();
    int instr_count=0;
    while(it != bhir.instr_list.end())
    {
        const unsigned int ksize = bhir.kernel_list.size();

        //Start new kernel
        bh_ir_kernel kernel;
        kernel.add_instr(*it);

        //Add the instruction to the map
        index_map.insert(pair<pair<int,int>,int>(pair<int,int>(ksize, 0), instr_count++));

        //Add fusible instructions to the kernel
        int i=1;
        for(it=it+1; it != bhir.instr_list.end(); ++it, ++i)
        {
            if(kernel.fusible(*it))
            {
                kernel.add_instr(*it);
                index_map.insert(pair<pair<int,int>,int>(pair<int,int>(ksize, i), instr_count++));
            }
            else
                break;
        }
        bhir.kernel_list.push_back(kernel);
    }
//    bhir.pprint_kernels();

    if(not bhir.check_kernel_cycles(index_map))
        throw logic_error("Cyclic dependencies between the kernels in the BhIR!");
}

