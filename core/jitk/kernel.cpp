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

#include <cassert>

#include <jitk/kernel.hpp>

using namespace std;

namespace bohrium {
namespace jitk {


Kernel::Kernel(const Block &block) : block(block) {

    _useRandom = false;
    const set<bh_base *> temps = getAllTemps();
    for (const bh_instruction *instr: getAllInstr()) {
        if (instr->opcode == BH_RANDOM) {
            _useRandom = true;
        } else if (instr->opcode == BH_FREE) {
            _frees.insert(instr->operand[0].base);
        } else if (instr->opcode == BH_SYNC) {
            _syncs.insert(instr->operand[0].base);
        }
        // Find non-temporary arrays
        const int nop = bh_noperands(instr->opcode);
        for (int i = 0; i < nop; ++i) {
            const bh_view &v = instr->operand[i];
            if (not bh_is_constant(&v) and temps.find(v.base) == temps.end()) {
                if (std::find(_non_temps.begin(), _non_temps.end(), v.base) == _non_temps.end()) {
                    _non_temps.push_back(v.base);
                }
            }
        }
    }
}

} // jitk
} // bohrium
