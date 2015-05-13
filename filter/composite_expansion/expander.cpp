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
#include "expander.hpp"

using namespace std;

namespace bohrium {
namespace filter {
namespace composite {

const char Expander::TAG[] = "Expander";

Expander::Expander(void)
{

}

void Expander::expand(bh_ir& bhir)
{
    int end = bhir.instr_list.size();
    for(int idx=0; idx<end; ++idx) {
        bh_instruction& instr = bhir.instr_list[idx];
        int increase = 0;
        switch(instr.opcode) {

            case BH_MATMUL:
                end += expand_matmul(bhir, idx);
                end += increase;
                idx += increase;
                break;

            case BH_SIGN:
                increase = expand_sign(bhir, idx);
                end += increase;
                idx += increase;
                break;

            default:
                break;
        }
    }
}

int Expander::gc()
{
    return 0;
}

Expander::~Expander(void)
{
    for(std::vector<bh_base*>::iterator it=bases_.begin();  // Deallocate bases
        it!=bases_.end();
        ++it) {
        delete *it;
    }
}

}}}

