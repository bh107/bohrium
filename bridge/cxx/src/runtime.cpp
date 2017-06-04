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

#include <bhxx/runtime.hpp>

using namespace std;

namespace bhxx {


void Runtime::instr_list_append(bh_instruction &instr) {
    if (instr_list.size() > 1000) {
        flush();
    }
    instr_list.push_back(instr);
}


void Runtime::flush() {
    bh_ir bhir = bh_ir(instr_list.size(), &instr_list[0]);
    runtime.execute(&bhir);
    instr_list.clear();
    for(bh_base *base: free_list) {
        delete base;
    }
    free_list.clear();
}

} // namespace bhxx
