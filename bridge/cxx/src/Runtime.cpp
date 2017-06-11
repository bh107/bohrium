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
#include <iterator>

using namespace std;

namespace bhxx {

Runtime::Runtime()
      : config(-1),                                // stack level -1 is the bridge
        runtime(config.getChildLibraryPath(), 0),  // and child is stack level 0
        extmethod_next_opcode_id(BH_MAX_OPCODE_ID + 1) {}

void Runtime::enqueue(BhInstruction instr) {
    instr_list.push_back(std::move(instr));

    // We hard-code a kernel size threshold here.
    // NB: we HAVE to include the just enqueued instruction since it might be a BH_FREE,
    // which clears `bases_for_deletion`.
    if (instr_list.size() >= 1000) {
        flush();
    }
}

void Runtime::enqueue_random(BhArray<uint64_t>& out, uint64_t seed, uint64_t key) {
    BhInstruction instr(BH_RANDOM);
    instr.append_operand(out);  // Append output array

    // Append the special BH_R123 constant
    bh_constant cnt;
    cnt.type             = bh_type::R123;
    cnt.value.r123.start = seed;
    cnt.value.r123.key   = key;
    instr.append_operand(cnt);

    enqueue(std::move(instr));
}

void Runtime::enqueue_deletion(std::unique_ptr<BhBase> base_ptr) {
    // Check whether we are responsible for the memory or not.
    if (!base_ptr->own_memory()) {
        // Externally managed
        // => set it to null to avoid deletion by Bohrium
        base_ptr->data = nullptr;
    }

    BhInstruction instr(BH_FREE);
    instr.append_operand(*base_ptr);
    bases_for_deletion.push_back(std::move(base_ptr));
    enqueue(std::move(instr));
}

void Runtime::flush() {
    // Construct Bohrium Internal Representation
    // and fill it with our instructions and execute.
    bh_ir bhir;
    std::move(instr_list.begin(), instr_list.end(), std::back_inserter(bhir.instr_list));
    runtime.execute(&bhir);
    instr_list.clear();

    // Purge the bases we have scheduled for deletion:
    bases_for_deletion.clear();
}

}  // namespace bhxx
