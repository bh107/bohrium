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

#include <map>
#include <string>
#include <algorithm>
#include <tuple>
#include <iostream>
#include <sstream>
#include <numeric>

#include <bh_instruction.hpp>

using namespace std;

set<const bh_base *> bh_instruction::get_bases_const() const {
    set<const bh_base*> ret;
    int nop = bh_noperands(opcode);
    for(int o=0; o<nop; ++o) {
        const bh_view &view = operand[o];
        if (not bh_is_constant(&view))
            ret.insert(view.base);
    }
    return ret;
}

set<bh_base *> bh_instruction::get_bases() {
    set<bh_base*> ret;
    int nop = bh_noperands(opcode);
    for(int o=0; o<nop; ++o) {
        const bh_view &view = operand[o];
        if (not bh_is_constant(&view))
            ret.insert(view.base);
    }
    return ret;
}

vector<const bh_view*> bh_instruction::get_views() const {
    vector<const bh_view*> ret;
    int nop = bh_noperands(opcode);
    for(int o=0; o<nop; ++o) {
        const bh_view &view = operand[o];
        if (not bh_is_constant(&view))
            ret.push_back(&view);
    }
    return ret;
}

bool bh_instruction::is_contiguous() const {
    int nop = bh_noperands(opcode);
    for(int o=0; o<nop; ++o) {
        const bh_view &view = operand[o];
        if ((not bh_is_constant(&view)) and (not bh_is_contiguous(&view)))
            return false;
    }
    return true;
}

bool bh_instruction::all_same_shape() const {
    const int nop = bh_noperands(opcode);
    if (nop > 0) {
        assert(not bh_is_constant(&operand[0]));
        const bh_view &first = operand[0];
        for(int o=1; o<nop; ++o) {
            const bh_view &view = operand[o];
            if (not bh_is_constant(&view)) {
                if (not bh_view_same_shape(&first, &view))
                    return false;
            }
        }
    }
    return true;
}

bool bh_instruction::reshapable() const {
    // It is not meaningful to reshape instructions with different shaped views
    // and for now we cannot reshape non-contiguous or sweeping instructions
    return all_same_shape() and is_contiguous() and not bh_opcode_is_sweep(opcode);
}

vector<int64_t> bh_instruction::shape() const {
    if (bh_opcode_is_sweep(opcode)) {
        // The principal shape of a sweep is the shape of the array array that is sweeped over
        assert(bh_noperands(opcode) == 3);
        assert(bh_is_constant(&operand[2]));
        assert(not bh_is_constant(&operand[1]));
        const bh_view &view = operand[1];
        return vector<int64_t>(view.shape, view.shape + view.ndim);
    } else if (opcode == BH_GATHER) {
        // The principal shape of a gather is the shape of the index and output array, which are equal.
        assert(bh_noperands(opcode) == 3);
        assert(not bh_is_constant(&operand[1]));
        assert(not bh_is_constant(&operand[2]));
        const bh_view &view = operand[2];
        return vector<int64_t>(view.shape, view.shape + view.ndim);
    } else if (bh_noperands(opcode) == 0) {
        // The principal shape of an instruction with no operands is the empty list
        return vector<int64_t>();
    } else {
        // The principal shape of a default instruction is the shape of the output
        const bh_view &view = operand[0];
        return vector<int64_t>(view.shape, view.shape + view.ndim);
    }
}

int64_t bh_instruction::ndim() const {
    return shape().size();
}

int bh_instruction::sweep_axis() const {
    if (bh_opcode_is_sweep(opcode)) {
        assert(bh_noperands(opcode) == 3);
        assert(bh_is_constant(&operand[2]));
        return static_cast<int>(constant.get_int64());
    }
    return BH_MAXDIM;
}

void bh_instruction::reshape(const vector<int64_t> &shape) {
    if (not reshapable()) {
        throw runtime_error("Reshape: instruction not reshapable!");
    }
    const int64_t totalsize = std::accumulate(shape.begin(), shape.end(), int64_t{1}, std::multiplies<int64_t>());
    int nop = bh_noperands(opcode);
    for(int o=0; o<nop; ++o) {
        bh_view &view = operand[o];
        if (bh_is_constant(&view))
            continue;
        if (totalsize != bh_nelements(view)) {
            throw runtime_error("Reshape: shape mismatch!");
        }

        // Let's assign the new shape and stride
        view.ndim = shape.size();
        copy(shape.begin(), shape.end(), view.shape);
        bh_set_contiguous_stride(&view);
    }
}

void bh_instruction::reshape_force(const vector<int64_t> &shape) {
    int nop = bh_noperands(opcode);
    for(int o=0; o<nop; ++o) {
        bh_view &view = operand[o];
        if (bh_is_constant(&view))
            continue;
        // Let's assign the new shape and stride
        view.ndim = shape.size();
        copy(shape.begin(), shape.end(), view.shape);
        bh_set_contiguous_stride(&view);
    }
}

void bh_instruction::remove_axis(int64_t axis) {
    assert(0 <= axis and axis < ndim());
    int nop = bh_noperands(opcode);
    if (nop > 0) {
        // In the input we can simply remove the axis
        for(int o=1; o<nop; ++o) {
            if (not bh_is_constant(&operand[o])) {
                operand[o].remove_axis(axis);
            }
        }
        // We might have to correct the sweep axis
        const int sa = sweep_axis();
        if (sa == axis) {
            throw runtime_error("remove_axis(): cannot remove an axis that is sweeped");
        } else if (sa > axis and sa < BH_MAXDIM) {
            constant.set_double(sa-1);
        }
        // In the output, we might have to correct the axis
        bh_view &view = operand[0];
        if (bh_opcode_is_reduction(opcode)) {
            view.remove_axis(sa < axis ? axis - 1 : axis);
        } else {
            // Otherwise, we just do the transpose
            view.remove_axis(axis);
        }
    }
}

void bh_instruction::transpose(int64_t axis1, int64_t axis2) {
    assert(0 <= axis1 and axis1 < ndim());
    assert(0 <= axis2 and axis2 < ndim());
    assert(axis1 != axis2);
    int nop = bh_noperands(opcode);
    if (nop > 0) {
        // The input we can simply transpose
        for(int o=1; o<nop; ++o) {
            bh_view &view = operand[o];
            if (not bh_is_constant(&view)) {
                view.transpose(axis1, axis2);
            }
        }
        // In the output, we have to handle sweep operations specially
        bh_view &view = operand[0];
        // First, we might have to swap the sweep axis
        const int sa = sweep_axis();
        if (sa == axis1) {
            constant.set_double(axis2);
        } else if (sa == axis2) {
            constant.set_double(axis1);
        }
        // Swapping a reduction means we might have to correct 'axis1' or 'axis2'
        if (bh_opcode_is_reduction(opcode)) {
            // But if we are reducing one of the swapped axes, the output shouldn't be transposed at all
            if (sa != axis1 and sa != axis2) {
                const int64_t t1 = sa<axis1?axis1-1:axis1;
                const int64_t t2 = sa<axis2?axis2-1:axis2;
                assert(t1 != t2);
                view.transpose(t1, t2);
            }
        } else {
            // Otherwise, we just do the transpose
            view.transpose(axis1, axis2);
        }
    }
}

bh_type bh_instruction::operand_type(int operand_index) const {
    assert(bh_noperands(opcode) > operand_index);
    const bh_view &view = operand[operand_index];
    if (bh_is_constant(&view)) {
        return constant.type;
    } else {
        return view.base->type;
    }
}

string bh_instruction::pprint(bool python_notation) const {
    stringstream ss;
    if(opcode > BH_MAX_OPCODE_ID)//It is an extension method
        ss << "ExtMethod";
    else//Regular instruction
        ss << bh_opcode_text(opcode);

    for(int i=0; i < bh_noperands(opcode); i++)
    {
        const bh_view &v = operand[i];
        ss << " ";
        if(bh_is_constant(&v)) {
            ss << constant;
        } else {
            ss << v.pprint(python_notation);
        }
    }
    return ss.str();
}

//Implements pprint of an instruction
ostream& operator<<(ostream& out, const bh_instruction& instr)
{
    out << instr.pprint(true);
    return out;
}

/* Retrieve the operands of a instruction.
 *
 * @instruction  The instruction in question
 * @return The operand list
 */
bh_view *bh_inst_operands(bh_instruction *instruction)
{
    return (bh_view *) &instruction->operand;
}

/* Determines whether instruction 'a' depends on instruction 'b',
 * which is true when:
 *      'b' writes to an array that 'a' access
 *                        or
 *      'a' writes to an array that 'b' access
 *
 * @a The first instruction
 * @b The second instruction
 * @return The boolean answer
 */
bool bh_instr_dependency(const bh_instruction *a, const bh_instruction *b)
{
    const int a_nop = bh_noperands(a->opcode);
    const int b_nop = bh_noperands(b->opcode);
    if(a_nop == 0 or b_nop == 0)
        return false;
    for(int i=0; i<a_nop; ++i)
    {
        if(not bh_view_disjoint(&b->operand[0], &a->operand[i]))
            return true;
    }
    for(int i=0; i<b_nop; ++i)
    {
        if(not bh_view_disjoint(&a->operand[0], &b->operand[i]))
            return true;
    }
    return false;
}
