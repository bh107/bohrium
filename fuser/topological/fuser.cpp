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
    uint64_t idx=0;
    while(idx < bhir.instr_list.size())
    {
        //Start new kernel
        bh_ir_kernel kernel(bhir);
        kernel.add_instr(idx);

        //Add fusible instructions to the kernel
        for(idx=idx+1; idx< bhir.instr_list.size(); ++idx)
        {
            //Check that 'it' is fusible with all instructions 'kernel'
            bool fusible = true;
            // TODO: Change needed when changing representation of kernel-instructions
            BOOST_FOREACH(uint64_t k_idx, kernel.instr_indexes)
            {
                if(not bohrium::check_fusible(&bhir.instr_list[idx], &bhir.instr_list[k_idx]))
                {
                    fusible = false;
                    break;
                }
            }
            if(fusible)
            {
                kernel.add_instr(idx);
            }
            else
                break;
        }
        bhir.kernel_list.push_back(kernel);
    }
}

