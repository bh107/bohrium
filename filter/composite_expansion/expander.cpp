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
#include "expander.hpp"

using namespace std;

namespace bohrium {
namespace filter {
namespace composite {

const char Expander::TAG[] = "Expander";

Expander::Expander(size_t threshold, int matmul, int sign)
    : gc_threshold_(threshold), matmul_(matmul), sign_(sign) {}

void Expander::expand(bh_ir& bhir)
{
    int end = bhir.instr_list.size();
    for(int pc=0; pc<end; ++pc) {
        bh_instruction& instr = bhir.instr_list[pc];
        int increase = 0;
        switch(instr.opcode) {

            case BH_MATMUL:
                end += expand_matmul(bhir, pc);
                end += increase;
                pc += increase;
                break;

            case BH_SIGN:
                increase = expand_sign(bhir, pc);
                end += increase;
                pc += increase;
                break;

            default:
                break;
        }
    }
}

size_t Expander::gc(void)
{
    size_t collected = 0;
    size_t size = bases_.size();

    if ((gc_threshold_) and (size>gc_threshold_)) {
        for(size_t limit = size-gc_threshold_;
            collected < limit;
            ++collected) {
            delete bases_.back();
            bases_.pop_back();
        }
    }
    return collected;
}

bh_base* Expander::make_base(bh_type type, bh_index nelem)
{
    bh_base* base = NULL;
    try {
        base = new bh_base;
    } catch (std::bad_alloc& ba) {
        base = NULL;
        fprintf(stderr, "Expander::make_base(...) bh_base allocation failed.\n");
        throw std::runtime_error("Expander::make_base(...) bh_base allocation failed.\n");
    }
    
    base->type = type;
    base->nelem = nelem;
    base->data = NULL;
    bases_.push_back(base);

    return base;
}

bh_view Expander::make_temp(bh_view& meta, bh_type type, bh_index nelem)
{
    bh_view view = meta;
    view.base = make_base(type, nelem);
    return view;
}

void Expander::inject(bh_ir& bhir, int pc, bh_opcode opcode, bh_view& out)
{
    bh_instruction instr;
    instr.opcode = opcode;
    instr.operand[0] = out;
    bhir.instr_list.insert(bhir.instr_list.begin()+pc, instr);
}

void Expander::inject(bh_ir& bhir, int pc, bh_opcode opcode, bh_view& out, bh_view& in1)
{
    bh_instruction instr;
    instr.opcode = opcode;
    instr.operand[0] = out;
    instr.operand[1] = in1;
    bhir.instr_list.insert(bhir.instr_list.begin()+pc, instr);
}

Expander::~Expander(void)
{
    while(bases_.size()>0) {
        delete bases_.back();
        bases_.pop_back();
    }
}

}}}

