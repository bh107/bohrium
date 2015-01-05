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
#include <bh_fuse.h>
#include <boost/foreach.hpp>
#include <vector>

using namespace std;
using namespace boost;

void fuser(bh_ir &bhir)
{
    if(bhir.kernel_list.size() != 0)
    {
        throw logic_error("The kernel_list is not empty!");
    }
    vector<bh_instruction>::iterator it = bhir.instr_list.begin();
    while(it != bhir.instr_list.end())
    {
        //Start new kernel
        bh_ir_kernel kernel;
        kernel.add_instr(*it);

        //Add fusible instructions to the kernel
        for(it=it+1; it != bhir.instr_list.end(); ++it)
        {
            //Check that 'it' is fusible with all instructions 'kernel'
            bool fusible = true;
            // TODO: Change needed when changing representation of kernel-instructions
            BOOST_FOREACH(const bh_instruction &i, kernel.get_instrs())
            {
                if(not bohrium::check_fusible(&i, &(*it)))
                {
                    fusible = false;
                    break;
                }
            }
            if(fusible)
            {
                kernel.add_instr(*it);
            }
            else
                break;
        }
        bhir.kernel_list.push_back(kernel);
    }
}

