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

#include <bh_component.hpp>
#include <bh_fuser.hpp>

using namespace bohrium;
using namespace component;
using namespace std;

class Impl : public ComponentFuser {
public:
    Impl(unsigned int stack_level) : ComponentFuser(stack_level) {}
    ~Impl() {};
    void do_fusion(bh_ir &bhir) {
        uint64_t idx=0;
        while(idx < bhir.instr_list.size())
        {
            //Start new kernel
            bh_ir_kernel kernel(bhir);
            kernel.add_instr(idx);

            //Add fusible instructions to the kernel
            for(idx=idx+1; idx < bhir.instr_list.size(); ++idx)
            {
                if(kernel.fusible(idx))
                {
                    kernel.add_instr(idx);
                }
                else
                    break;
            }
            bhir.kernel_list.push_back(kernel);
        }
    }
};

extern "C" ComponentImpl* create(unsigned int stack_level) {
    return new Impl(stack_level);
}
extern "C" void destroy(ComponentImpl* self) {
    delete self;
}