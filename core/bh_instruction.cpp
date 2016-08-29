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

#include <bh_instruction.hpp>

using namespace std;

set<bh_base*> bh_instruction::get_bases() {
    set<bh_base*> ret;
    int nop = bh_noperands(opcode);
    for(int o=0; o<nop; ++o) {
        const bh_view &view = operand[o];
        if (not bh_is_constant(&view))
            ret.insert(view.base);
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

bool bh_instruction::reshapable() const {
   return (bh_opcode_is_elementwise(opcode) or bh_opcode_is_system(opcode)) and is_contiguous();
}

int64_t bh_instruction::max_ndim() const {
    // Let's find the view with the greatest number of dimension and returns its 'ndim'
    int64_t ret = 0;
    int nop = bh_noperands(opcode);
    for(int o=0; o<nop; ++o) {
        const bh_view &view = operand[o];
        if (not bh_is_constant(&view)) {
            if (view.ndim > ret)
                ret = view.ndim;
        }
    }
    return ret;
}

vector<int64_t> bh_instruction::dominating_shape() const {
    int64_t ndim = max_ndim();
    vector<int64_t > shape;
    int nop = bh_noperands(opcode);
    for(int o=0; o<nop; ++o) {
        const bh_view &view = operand[o];
        if ((not bh_is_constant(&view)) and view.ndim == ndim) {
            for (int64_t j=0; j < view.ndim; ++j) {
                if (shape.size() > (size_t)j) {
                    if (shape[j] < view.shape[j])
                        shape[j] = view.shape[j];
                } else {
                    shape.push_back(view.shape[j]);
                }
            }
        }
    }
    return shape;
}

void bh_instruction::reshape(const vector<int64_t> &shape) {
    if (not reshapable()) {
        throw runtime_error("Reshape: instruction not reshapable!");
    }
    int64_t totalsize = 1;
    for (int64_t dim: shape) {
        totalsize *= dim;
    }

    int nop = bh_noperands(opcode);
    for(int o=0; o<nop; ++o) {
        bh_view &view = operand[o];
        if (bh_is_constant(&view))
            continue;
        if (totalsize != bh_nelements(view)) {
            throw runtime_error("Reshape: shape mismatch!");
        }

        // Let's assign the new shape and strides
        view.ndim = shape.size();
        copy(shape.begin(), shape.end(), view.shape);
        bh_set_contiguous_stride(&view);
    }
}

//Implements pprint of an instruction
ostream& operator<<(ostream& out, const bh_instruction& instr)
{
    string name;
    if(instr.opcode > BH_MAX_OPCODE_ID)//It is an extension method
        name = "ExtMethod";
    else//Regular instruction
        name = bh_opcode_text(instr.opcode);
    out << name;

    for(int i=0; i < bh_noperands(instr.opcode); i++)
    {
        const bh_view &v = instr.operand[i];
        out << " ";
        if(bh_is_constant(&v))
            out << instr.constant;
        else
            out << v;
    }
    return out;
}

/* Retrive the operands of a instruction.
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
