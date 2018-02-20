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

#include <bhxx/BhInstruction.hpp>

namespace bhxx {

void BhInstruction::appendOperand(bh_constant cnt) {
    bh_view view;
    view.base = nullptr;
    operand.push_back(view);
    constant = cnt;
}

void BhInstruction::appendOperand(BhBase& base) {
    if (opcode != BH_FREE) {
        throw std::runtime_error(
              "BhBase objects can only be freed. Use a full BhArray if you want to "
              "berform any other operation on it.");
    }

    // Make a bh_view to this base
    bh_view view;
    view.base      = &base;
    view.start     = 0;
    view.ndim      = 1;
    view.shape[0]  = base.nelem;
    view.stride[0] = 1;

    operand.push_back(view);
}

template <typename T>
void BhInstruction::appendOperand(BhArray<T>& ary) {
    if (opcode == BH_FREE) {
        throw std::runtime_error(
              "BH_FREE cannot be used as an instruction on arrays in the bhxx interface. "
              "Use Runtime::instance().enqueue(BH_FREE,array) instead.");
    }

    // Forward to the const version below.
    appendOperand(const_cast<const BhArray<T>&>(ary));
}

template <typename T>
void BhInstruction::appendOperand(const BhArray<T>& ary) {
    if (opcode == BH_FREE) {
        throw std::runtime_error(
              "BH_FREE cannot be used as an instruction on arrays in the bhxx interface. "
              "Use Runtime::instance().enqueue(BH_FREE,array) instead.");
    }

    bh_view view;
    assert(ary.base.use_count() > 0);
    view.base  = ary.base.get();
    view.start = static_cast<int64_t>(ary.offset);
    view.ndim  = static_cast<int64_t>(ary.shape.size());
    view.dyn_dimensions = ary.dyn_dimensions;
    view.dyn_offsets = ary.dyn_offsets;

    std::copy(ary.shape.begin(), ary.shape.end(), &view.shape[0]);
    std::copy(ary.stride.begin(), ary.stride.end(), &view.stride[0]);
    operand.push_back(view);

}

template <typename T>
void BhInstruction::appendOperand(T scalar) {
    bh_view view;
    view.base = nullptr;
    operand.push_back(view);
    constant = bh_constant(scalar);
}

// Instantiate all possible types for bh_instruction
#define INSTANTIATE(TYPE)                                              \
    template void BhInstruction::appendOperand(TYPE);                 \
    template void BhInstruction::appendOperand(const BhArray<TYPE>&); \
    template void BhInstruction::appendOperand(BhArray<TYPE>&)

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
