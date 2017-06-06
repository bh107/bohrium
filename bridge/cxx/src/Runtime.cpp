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

#include <bhxx/Runtime.hpp>

using namespace std;

namespace bhxx {

Runtime::Runtime()
      : config(-1),                                // stack level -1 is the bridge
        runtime(config.getChildLibraryPath(), 0),  // and child is stack level 0
        extmethod_next_opcode_id(BH_MAX_OPCODE_ID + 1) {}

void Runtime::enqueue(BhInstruction instr) {
    if (instr_list.size() > 1000) flush();
    instr_list.push_back(std::move(instr));
}

void Runtime::enqueue_random(BhArray<uint64_t>& out, uint64_t seed, uint64_t key) {
    BhInstruction instr(BH_RANDOM);
    instr.append_operand(out);  // Append output array

    // Append the special BH_R123 constant
    bh_constant cnt;
    cnt.type             = BH_R123;
    cnt.value.r123.start = seed;
    cnt.value.r123.key   = key;
    instr.append_operand(cnt);

    enqueue(std::move(instr));
}

void Runtime::flush() {
    // Construct Bohrium Internal Representation
    // and fill it with our instructions.
    bh_ir bhir;
    bhir.instr_list.resize(instr_list.size());
    std::transform(
          instr_list.begin(), instr_list.end(), bhir.instr_list.begin(),
          [](const BhInstruction& bi) { return static_cast<const bh_instruction&>(bi); });

    // Execute them
    runtime.execute(&bhir);
    instr_list.clear();
}

}  // namespace bhxx
