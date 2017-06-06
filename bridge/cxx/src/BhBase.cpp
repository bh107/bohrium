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

#include <bh_memory.h>
#include <bhxx/BhBase.hpp>
#include <bhxx/BhInstruction.hpp>
#include <bhxx/Runtime.hpp>

// TODO get rid of this header
//      (Still from the old interface)
#include <bxx/traits.hpp>

namespace bhxx {

template <typename T>
void BhBase::set_type() {
    bxx::assign_array_type<T>(this);
}

void BhBaseDeleter::operator()(BhBase* ptr) const {
    // Check whether we are responsible for the memory or not.
    if (!ptr->own_memory()) {
        // Externally managed -> set it to null to avoid
        // deletion by Bohrium
        ptr->data = nullptr;
    }

    // Now send BH_FREE to Bohrium and hand the
    // ownership of the BhBase object over to
    // the BhInstruction object by the means of a unique_ptr
    BhInstruction instr(BH_FREE);
    instr.append_operand(std::unique_ptr<BhBase>(ptr));
    Runtime::instance().enqueue(std::move(instr));
}

// Instantiate all possible types of the set_type function
#define INSTANTIATE(TYPE) template void BhBase::set_type<TYPE>()

INSTANTIATE(bool);
INSTANTIATE(int8_t);
INSTANTIATE(int16_t);
INSTANTIATE(int32_t);
INSTANTIATE(int64_t);
INSTANTIATE(uint8_t);
INSTANTIATE(uint16_t);
INSTANTIATE(uint32_t);
INSTANTIATE(uint64_t);
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(std::complex<float>);
INSTANTIATE(std::complex<double>);

#undef INSTANTIATE
}
