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
#include <numeric>
#include <chrono>

#include <bh_component.hpp>
#include <bh_extmethod.hpp>
#include <bh_util.hpp>
#include <bh_opcode.h>
#include <jitk/fuser.hpp>
#include <jitk/block.hpp>
#include <jitk/instruction.hpp>
#include <jitk/graph.hpp>
#include <jitk/transformer.hpp>
#include <jitk/fuser_cache.hpp>
#include <jitk/codegen_util.hpp>
#include <jitk/statistics.hpp>
#include <jitk/dtype.hpp>
#include <jitk/apply_fusion.hpp>
#include <jitk/engines/dyn_view.hpp>

#include "engine_pgas.hpp"

using namespace bohrium;
using namespace jitk;
using namespace component;
using namespace std;

namespace {
class Impl : public ComponentVE {
  private:
    //Allocated base arrays
    set<bh_base*> _allocated_bases;

  public:
    // Some statistics
    Statistics stat;
    // The OpenMP engine
    EnginePGAS engine;

    Impl(int stack_level) : ComponentVE(stack_level), stat(config), engine(*this, stat) {}
    ~Impl() override;
    void execute(BhIR *bhir) override;
    void extmethod(const string &name, bh_opcode opcode) override {
        // ExtmethodFace does not have a default or copy constructor thus
        // we have to use its move constructor.
        extmethods.insert(make_pair(opcode, extmethod::ExtmethodFace(config, name)));
    }

    // Handle messages from parent
    string message(const string &msg) override {
        stringstream ss;
        if (msg == "statistic_enable_and_reset") {
            stat = Statistics(true, config);
        } else if (msg == "statistic") {
            engine.updateFinalStatistics();
            stat.write("PGAS", "", ss);
            return ss.str();
        } else if (msg == "info") {
            ss << engine.info();
        }
        return ss.str();
    }

    // Handle memory pointer retrieval
    void* getMemoryPointer(bh_base &base, bool copy2host, bool force_alloc, bool nullify) override {
        if (not copy2host) {
            throw runtime_error("PGAS - getMemoryPointer(): `copy2host` is not True");
        }
        if (force_alloc) {
            bh_data_malloc(&base);
        }
        void *ret = base.getDataPtr();
        if (nullify) {
            base.resetDataPtr();
        }
        return ret;
    }

    // Handle memory pointer obtainment
    void setMemoryPointer(bh_base *base, bool host_ptr, void *mem) override {
        if (not host_ptr) {
            throw runtime_error("PGAS - setMemoryPointer(): `host_ptr` is not True");
        }
        if (base->getDataPtr() != nullptr) {
            throw runtime_error("PGAS - setMemoryPointer(): `base->getDataPtr()` is not NULL");
        }
        base->resetDataPtr(mem);
    }

    // We have no context so returning NULL
    void* getDeviceContext() override {
        return nullptr;
    };

    // We have no context so doing nothing
    void setDeviceContext(void* device_context) override {};
};
}

extern "C" ComponentImpl* create(int stack_level) {
    return new Impl(stack_level);
}
extern "C" void destroy(ComponentImpl* self) {
    delete self;
}

Impl::~Impl() {
    if (stat.print_on_exit) {
        engine.updateFinalStatistics();
        stat.write("PGAS", config.defaultGet<std::string>("prof_filename", ""), cout);
    }
}

namespace {
// Return the first base in `bhir` or nullptr
bh_base *get_first_base(BhIR *bhir) {
    for (const bh_instruction &instr: bhir->instr_list) {
        for (const bh_view &view: instr.getViews()) {
            return view.base;
        }
    }
    return nullptr;
}
}

void Impl::execute(BhIR *bhir) {
    // Let's send regular arrays to the child and check that pgas and regular arrays are not mixed
    {
        bh_base *base = get_first_base(bhir);
        if (base != nullptr) {
            if (not base->pgas.enabled()) {
                // Let's make sure that ALL bases are non-pgas arrays
                for (const bh_instruction &instr: bhir->instr_list) {
                    for (const bh_view &view: instr.getViews()) {
                        if (view.base->pgas.enabled()) {
                            throw std::runtime_error("PGAS: cannot mix pgas and regular arrays");
                        }
                    }
                }
                child.execute(bhir);
                return;
            } else {
                // Let's make sure that ALL bases are pgas arrays
                for (const bh_instruction &instr: bhir->instr_list) {
                    for (const bh_view &view: instr.getViews()) {
                        if (not view.base->pgas.enabled()) {
                            throw std::runtime_error("PGAS: cannot mix pgas and regular arrays");
                        }
                    }
                }
            }
        }
    }

    bh_base *cond = bhir->getRepeatCondition();
    for (uint64_t i = 0; i < bhir->getNRepeats(); ++i) {
        // Let's handle extension methods
        engine.handleExtmethod(bhir);

        // And then the regular instructions
        engine.handleExecution(bhir);

        // Check condition
        if (cond != nullptr and cond->getDataPtr() != nullptr and not((bool *) cond->getDataPtr())[0]) {
            break;
        }

        // Change views that slide between iterations
        slide_views(bhir);
    }
}
