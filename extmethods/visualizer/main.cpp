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
#include <cassert>
#include <iostream>
#include <bh_extmethod.hpp>
#include "visualizer.hpp"

using namespace bohrium;
using namespace extmethod;
using namespace std;

namespace {
class VisualizerImpl : public ExtmethodImpl {
private:
    bool bh_visualize_initialized = false;

public:
    void execute(bh_instruction *instr, void* arg) override {
        bh_view *subject = &instr->operand[0];
        bh_float32 *args = reinterpret_cast<bh_float32 *>(instr->operand[1].base->data);
        assert(args != nullptr);
        assert(instr->operand[1].base->nelem == 5);
        assert(instr->operand[1].base->type == bh_type::FLOAT32);

        for(int64_t i=0; i<subject->ndim; ++i) {
            if(subject->shape[i] < 16) {
                throw runtime_error("Cannot visualize because of input shape");
            }
        }
        if (subject->base->data == nullptr) {
            throw runtime_error("You are trying to visualize non-existing data");
        }

        bh_int32 cm    = static_cast<bh_int32>(args[0]);
        bh_bool flat   = static_cast<bh_bool>(args[1]);
        bh_bool cube   = static_cast<bh_bool>(args[2]);
        bh_float32 min = static_cast<bh_float32>(args[3]);
        bh_float32 max = static_cast<bh_float32>(args[4]);

        if (!bh_visualize_initialized) {
            if (subject->ndim == 3) {
                Visualizer::getInstance().setValues(
                        subject,
                        static_cast<int>(subject->shape[0]),
                        static_cast<int>(subject->shape[1]),
                        static_cast<int>(subject->shape[2]),
                        cm, flat, cube, min, max
                );
            } else {
                Visualizer::getInstance().setValues(
                        subject,
                        static_cast<int>(subject->shape[0]),
                        static_cast<int>(subject->shape[1]),
                        1,
                        cm, flat, cube, min, max
                );
            }
            bh_visualize_initialized = true;
        }
        Visualizer::getInstance().run(subject);
    }
};
}

extern "C" ExtmethodImpl* visualizer_create() {
    return new VisualizerImpl();
}
extern "C" void visualizer_destroy(ExtmethodImpl* self) {
    delete self;
}
