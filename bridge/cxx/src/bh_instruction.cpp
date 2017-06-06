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

#include <bhxx/bh_instruction.hpp>
#include <bxx/traits.hpp>

namespace bhxx {

void BhInstruction::append_operand(bh_constant cnt) {
    bh_view view;
    view.base = nullptr;
    operand.push_back(view);
    constant = cnt;
}

void BhInstruction::append_operand(BhBase base) {
    if (opcode != BH_FREE) {
        throw std::runtime_error(
              "BhBase objects can only be freed. Use a full BhArray if you want to call "
              "any other operation on it.");
    }

    // Move the data to the unique pointer, which
    // will delete it once this object is deleted.
    base_ptr.reset(new BhBase(std::move(base)));

    // Make a bh_view to this base
    bh_view view;
    view.base      = base_ptr.get();
    view.start     = 0;
    view.ndim      = 1;
    view.shape[0]  = base_ptr->nelem;
    view.stride[0] = 1;

    operand.push_back(view);
}

template <typename T>
void BhInstruction::append_operand(BhArray<T>& ary) {
    if (opcode == BH_FREE) {
        // Calling BH_FREE on an array with external
        // storage management is undefined behaviour
        if (!ary.base->own_memory()) {
            throw std::runtime_error(
                  "Cannot call BH_FREE on a BhArray object, which reference a BhBase "
                  "object, which uses external storage.");
        }

        // BH_FREE  is special because it is automatically invoked
        // by the deleter of the shared pointer to BhBase if that
        // thing is not used any more. So we just free the reference
        // to the BhBase.
        ary.base.reset();
    } else {
        // Forward to the const version below.
        append_operand(const_cast<const BhArray<T>&>(ary));
    }
}

template <typename T>
void BhInstruction::append_operand(const BhArray<T>& ary) {
    if (opcode == BH_FREE) {
        throw std::runtime_error("Const arrays cannot be freed.");
    }

    bh_view view;
    view.base  = ary.base.get();
    view.start = static_cast<bh_index>(ary.offset);
    view.ndim  = static_cast<bh_intp>(ary.shape.size());
    std::copy(ary.shape.begin(), ary.shape.end(), &view.shape[0]);
    std::copy(ary.stride.begin(), ary.stride.end(), &view.stride[0]);
    operand.push_back(view);
}

template <typename T>
void BhInstruction::append_operand(T scalar) {
    bh_view view;
    view.base = nullptr;
    operand.push_back(view);
    bxx::assign_const_type(&constant, scalar);
}

// Instantiate all possible types for bh_instruction
#define INSTANTIATE(TYPE)                                              \
    template void BhInstruction::append_operand(TYPE);                 \
    template void BhInstruction::append_operand(const BhArray<TYPE>&); \
    template void BhInstruction::append_operand(BhArray<TYPE>&)

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

}  // namespace bhxx
