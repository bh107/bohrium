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
namespace bcexp {
    bool __verbose = false;

Expander::Expander(
    bool verbose,
    size_t threshold,
    int sign,
    int powk,
    int reduce1d)
    : gc_threshold_(threshold),
      sign_(sign),
      powk_(powk),
      reduce1d_(reduce1d) {
          __verbose = verbose;
      }

void Expander::expand(BhIR& bhir)
{
    int end = bhir.instr_list.size();
    for(int pc=0; pc<end; ++pc) {
        bh_instruction& instr = bhir.instr_list[pc];
        int increase = 0;
        switch(instr.opcode) {

        case BH_POWER:
            if (powk_) {
                increase = expandPowk(bhir, pc);
                end += increase;
                pc += increase;
            }
            break;

        case BH_SIGN:
            if (sign_) {
                increase = expandSign(bhir, pc);
                end += increase;
                pc += increase;
            }
            break;

        case BH_ADD_REDUCE:
        case BH_MULTIPLY_REDUCE:
        case BH_MINIMUM_REDUCE:
        case BH_MAXIMUM_REDUCE:
        case BH_LOGICAL_AND_REDUCE:
        case BH_BITWISE_AND_REDUCE:
        case BH_LOGICAL_OR_REDUCE:
        case BH_BITWISE_OR_REDUCE:
        case BH_LOGICAL_XOR_REDUCE:
        case BH_BITWISE_XOR_REDUCE:
            if (reduce1d_ && instr.operand[1].ndim == 1)
            {
                increase = expandReduce1d(bhir, pc, reduce1d_);
                end += increase;
                pc += increase;
            }
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

bh_base * Expander::createBase(bh_type type, int64_t nelem, const BhPGAS &pgas)
{
    bh_base* base = nullptr;
    try {
        base = new bh_base(nelem, type, pgas);
    } catch (std::bad_alloc& ba) {
        fprintf(stderr, "Expander::createBase(...) bh_base allocation failed.\n");
        throw std::runtime_error("Expander::createBase(...) bh_base allocation failed.\n");
    }
    bases_.push_back(base);
    return base;
}

bh_view Expander::createTemp(bh_view& meta, bh_type type, int64_t nelem)
{
    bh_view view = meta;
    view.base = createBase(type, nelem, meta.base->pgas);
    return view;
}

bh_view Expander::createTemp(bh_type type, int64_t nelem, const BhPGAS &pgas)
{
    bh_view view;
    view.base = createBase(type, nelem, pgas);
    view.start = 0;
    view.ndim = 1;
    view.shape.push_back(nelem);
    view.stride.push_back(1);
    return view;
}

void Expander::inject(BhIR& bhir, int pc, bh_opcode opcode, bh_view& out)
{
    bh_instruction instr(opcode, {out});
    bhir.instr_list.insert(bhir.instr_list.begin()+pc, instr);
}

void Expander::inject(BhIR& bhir, int pc, bh_opcode opcode, bh_view& out, bh_view& in1)
{
    bh_instruction instr(opcode, {out, in1});
    bhir.instr_list.insert(bhir.instr_list.begin()+pc, instr);
}

Expander::~Expander(void)
{
    while(bases_.size()>0) {
        delete bases_.back();
        bases_.pop_back();
    }
}

void verbose_print(std::string str)
{
    if (__verbose) {
        std::cout << "[Expander] " << str << std::endl;
    }
}

}}}
