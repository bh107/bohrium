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
    set<const bh_base *> ret;
    for (const bh_view &view: operand) {
        if (not bh_is_constant(&view))
            ret.insert(view.base);
    }
    return ret;
}

set<bh_base *> bh_instruction::get_bases() {
    set<bh_base *> ret;
    for (const bh_view &view: operand) {
        if (not bh_is_constant(&view))
            ret.insert(view.base);
    }
    return ret;
}

vector<const bh_view *> bh_instruction::get_views() const {
    vector<const bh_view *> ret;
    for (const bh_view &view: operand) {
        if (not bh_is_constant(&view))
            ret.push_back(&view);
    }
    return ret;
}

bool bh_instruction::isContiguous() const {
    for (const bh_view &view: operand) {
        if ((not bh_is_constant(&view)) and (not bh_is_contiguous(&view)))
            return false;
    }
    return true;
}

bool bh_instruction::all_same_shape() const {
    if (operand.size() > 0) {
        assert(not bh_is_constant(&operand[0]));
        const bh_view &first = operand[0];
        for (size_t o = 1; o < operand.size(); ++o) {
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
    return all_same_shape() and isContiguous() and not bh_opcode_is_sweep(opcode);
}

vector<int64_t> bh_instruction::shape() const {
    if (bh_opcode_is_sweep(opcode)) {
        // The principal shape of a sweep is the shape of the array array that is sweeped over
        assert(operand.size() == 3);
        assert(bh_is_constant(&operand[2]));
        assert(not bh_is_constant(&operand[1]));
        const bh_view &view = operand[1];
        return vector<int64_t>(view.shape, view.shape + view.ndim);
    } else if (opcode == BH_GATHER) {
        // The principal shape of a gather is the shape of the index and output array, which are equal.
        assert(operand.size() == 3);
        assert(not bh_is_constant(&operand[1]));
        assert(not bh_is_constant(&operand[2]));
        const bh_view &view = operand[2];
        return vector<int64_t>(view.shape, view.shape + view.ndim);
    } else if (opcode == BH_SCATTER or opcode == BH_COND_SCATTER) {
        // The principal shape of a scatter is the shape of the index and input array, which are equal.
        assert(operand.size() >= 3);
        assert(not bh_is_constant(&operand[1]));
        assert(not bh_is_constant(&operand[2]));
        const bh_view &view = operand[2];
        return vector<int64_t>(view.shape, view.shape + view.ndim);
    } else if (operand.empty()) {
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
        assert(operand.size() == 3);
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
    for (bh_view &view: operand) {
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
    for (bh_view &view: operand) {
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
    if (operand.size() > 0) {
        // In the input we can simply remove the axis
        for (size_t o = 1; o < operand.size(); ++o) {
            if (not(bh_is_constant(&operand[o]) or     // Ignore constants
                    (o == 1 and opcode == BH_GATHER))) // Ignore gather's first input operand
            {
                operand[o].remove_axis(axis);
            }
        }
        // We might have to correct the sweep axis
        const int sa = sweep_axis();
        if (sa == axis) {
            throw runtime_error("remove_axis(): cannot remove an axis that is sweeped");
        } else if (sa > axis and sa < BH_MAXDIM) {
            constant.set_double(sa - 1);
        }

        // Ignore scatter's output operand, which is allowed any shape
        if (opcode == BH_SCATTER or opcode == BH_COND_SCATTER) {
            return;
        }

        // In the output, we might have to correct the axis
        bh_view &view = operand[0];
        if (bh_opcode_is_reduction(opcode)) {
            view.remove_axis(sa < axis ? axis - 1 : axis);
        } else {
            // Otherwise, we just do the removal
            view.remove_axis(axis);
        }
    }
}

void bh_instruction::transpose(int64_t axis1, int64_t axis2) {
    assert(0 <= axis1 and axis1 < ndim());
    assert(0 <= axis2 and axis2 < ndim());
    assert(axis1 != axis2);
    if (not operand.empty()) {
        // The input we can simply transpose
        for (size_t o = 1; o < operand.size(); ++o) {
            bh_view &view = operand[o];
            if (not bh_is_constant(&view)) {
                if (not (o == 1 and opcode == BH_GATHER)) { // The input array of gather has arbitrary shape and stride
                    view.transpose(axis1, axis2);
                }
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

        // The output array of scatter is has arbitrary shape and stride
        if (opcode == BH_SCATTER or opcode == BH_COND_SCATTER) {
            return;
        }

        // Swapping a reduction means we might have to correct 'axis1' or 'axis2'
        if (bh_opcode_is_reduction(opcode)) {
            if (sa != axis1 and sa != axis2) {
                const int64_t t1 = sa < axis1 ? axis1 - 1 : axis1;
                const int64_t t2 = sa < axis2 ? axis2 - 1 : axis2;
                assert(t1 != t2);
                view.transpose(t1, t2);
            } else {
                // If we are reducing one of the swapped axes, we insert a dummy dimension into the output,
                // do the transpose, and than remove the dummy dimension again.
                if (sa != axis1) { // Make sure that `axis1` is the reduced dimension
                    std::swap(axis1, axis2);
                }
                view.insert_axis(axis1, 1, 1); // Dummy axis at the reduced dimension
                view.transpose(axis1, axis2);
                view.remove_axis(axis2); // Notice, now `axis2` is the reduced dimension
            }
        } else {
            // Otherwise, we just do the transpose
            view.transpose(axis1, axis2);
        }
    }
}

void bh_instruction::transpose() {
    int64_t nd = ndim();
    if (not operand.empty()) {
        int64_t lc = 0;
        int64_t rc = nd-1;
        while(lc < rc) {
            transpose(lc, rc);
            --rc;
            ++lc;
        }
    }
}

bh_type bh_instruction::operand_type(int operand_index) const {
    assert(((int) operand.size()) > operand_index);
    const bh_view &view = operand[operand_index];
    if (bh_is_constant(&view)) {
        return constant.type;
    } else {
        return view.base->type;
    }
}

string bh_instruction::pprint(bool python_notation) const {
    stringstream ss;
    if (opcode > BH_MAX_OPCODE_ID)//It is an extension method
        ss << "ExtMethod";
    else//Regular instruction
        ss << bh_opcode_text(opcode);

    for (const bh_view &v: operand) {
        ss << " ";
        if (bh_is_constant(&v)) {
            ss << constant;
        } else {
            ss << v.pprint(python_notation);
        }
    }
    return ss.str();
}

ostream &operator<<(ostream &out, const bh_instruction &instr) {
    out << instr.pprint(true);
    return out;
}

bool bh_instr_dependency(const bh_instruction *a, const bh_instruction *b) {
    const size_t a_nop = a->operand.size();
    const size_t b_nop = b->operand.size();
    if (a_nop == 0 or b_nop == 0)
        return false;
    for (size_t i = 0; i < a_nop; ++i) {
        if (not bh_view_disjoint(&b->operand[0], &a->operand[i]))
            return true;
    }
    for (size_t i = 0; i < b_nop; ++i) {
        if (not bh_view_disjoint(&a->operand[0], &b->operand[i]))
            return true;
    }
    return false;
}
