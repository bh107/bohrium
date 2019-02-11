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

#include <bohrium/bh_component.hpp>
#include <bohrium/bh_extmethod.hpp>
#include <bohrium/bh_util.hpp>
#include <bohrium/bh_opcode.h>
#include <bohrium/jitk/fuser.hpp>
#include <bohrium/jitk/block.hpp>
#include <bohrium/jitk/instruction.hpp>
#include <bohrium/jitk/graph.hpp>
#include <bohrium/jitk/transformer.hpp>
#include <bohrium/jitk/fuser_cache.hpp>
#include <bohrium/jitk/codegen_util.hpp>
#include <bohrium/jitk/statistics.hpp>
#include <bohrium/jitk/apply_fusion.hpp>
#include <bohrium/jitk/engines/dyn_view.hpp>

#include "engine_openmp.hpp"

using namespace bohrium;
using namespace jitk;
using namespace component;
using namespace std;

namespace {
class Impl : public ComponentVE {
private:
    //Allocated base arrays
    set<bh_base *> _allocated_bases;

public:
    // Some statistics
    Statistics stat;
    // The OpenMP engine
    EngineOpenMP engine;

    Impl(int stack_level) : ComponentVE(stack_level),
                            stat(config),
                            engine(*this, stat) {}

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
            stat.write("OpenMP", "", ss);
            return ss.str();
        } else if (msg == "info") {
            ss << engine.info();
        }
        return ss.str();
    }

    // Handle memory pointer retrieval
    void *getMemoryPointer(bh_base &base, bool copy2host, bool force_alloc, bool nullify) override {
        if (not copy2host) {
            throw runtime_error("OpenMP - getMemoryPointer(): `copy2host` is not True");
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
            throw runtime_error("OpenMP - setMemoryPointer(): `host_ptr` is not True");
        }
        if (base->getDataPtr() != nullptr) {
            throw runtime_error("OpenMP - setMemoryPointer(): `base->getDataPtr()` is not NULL");
        }
        base->resetDataPtr(mem);
    }

    // We have no context so returning NULL
    void *getDeviceContext() override {
        return nullptr;
    };

    // We have no context so doing nothing
    void setDeviceContext(void *device_context) override {};

    // Handle user kernels
    string userKernel(const std::string &kernel, std::vector<bh_view> &operand_list,
                      const std::string &compile_cmd, const std::string &tag, const std::string &param) override {
        if (tag == "openmp") {
            const auto texecution = chrono::steady_clock::now();
            string ret = engine.userKernel(kernel, operand_list, compile_cmd, tag, param);
            stat.time_total_execution += chrono::steady_clock::now() - texecution;
            return ret;
        } else {
            throw std::runtime_error("No backend with tag \"" + tag + "\" found");
        }
    }

};
}

extern "C" ComponentImpl *create(int stack_level) {
    return new Impl(stack_level);
}
extern "C" void destroy(ComponentImpl *self) {
    delete self;
}

Impl::~Impl() {
    if (stat.print_on_exit) {
        engine.updateFinalStatistics();
        stat.write("OpenMP", config.defaultGet<std::string>("prof_filename", ""), cout);
    }
}

void Impl::execute(BhIR *bhir) {
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
