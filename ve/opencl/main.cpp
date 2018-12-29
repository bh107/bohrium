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
#include <set>
#include <map>
#include <chrono>

#include <bh_component.hpp>
#include <bh_extmethod.hpp>
#include <bh_util.hpp>
#include <jitk/statistics.hpp>
#include <jitk/engines/dyn_view.hpp>

#include "engine_opencl.hpp"

using namespace bohrium;
using namespace jitk;
using namespace component;
using namespace std;

namespace {
class Impl : public ComponentVE {
  public:
    // Some statistics
    Statistics stat;
    // The OpenCL engine
    EngineOpenCL engine;
    // Optimize instruction to access column major
    const bool to_col_major;

    Impl(int stack_level) : ComponentVE(stack_level),
                            stat(config),
                            engine(*this, stat),
                            to_col_major(config.defaultGet<bool>("to_col_major", true)) {}
    ~Impl() override;
    void execute(BhIR *bhir) override;
    void extmethod(const string &name, bh_opcode opcode) override {
        // ExtmethodFace does not have a default or copy constructor thus
        // we have to use its move constructor.
        try {
            extmethods.insert(make_pair(opcode, extmethod::ExtmethodFace(config, name)));
        } catch(extmethod::ExtmethodNotFound &e) {
            // I don't know this function, lets try my child
            child.extmethod(name, opcode);
            child_extmethods.insert(opcode);
        }
    }

    // Handle messages from parent
    string message(const string &msg) override {
        stringstream ss;
        if (msg == "statistic_enable_and_reset") {
            stat = Statistics(true, config);
        } else if (msg == "statistic") {
            engine.updateFinalStatistics();
            stat.write("OpenCL", "", ss);
        } else if (msg == "GPU: disable") {
            engine.copyAllBasesToHost();
            disabled = true;
        } else if (msg == "GPU: enable") {
            disabled = false;
        } else if (msg == "info") {
            ss << engine.info();
        }
        return ss.str() + child.message(msg);
    }

    // Handle memory pointer retrieval
    void* getMemoryPointer(bh_base &base, bool copy2host, bool force_alloc, bool nullify) override {
        bh_base *b = &base;
        if (copy2host) {
            std::set<bh_base*> t = { b };
            engine.copyToHost(t);
            engine.delBuffer(b);
            if (force_alloc) {
                bh_data_malloc(b);
            }
            void *ret = base.getDataPtr();
            if (nullify) {
                base.resetDataPtr();
            }
            return ret;
        } else {
            return engine.getCBuffer(b);
        }
    }

    // Handle memory pointer obtainment
    void setMemoryPointer(bh_base *base, bool host_ptr, void *mem) override {
        if (host_ptr) {
            std::set<bh_base*> t = { base };
            engine.copyToHost(t);
            engine.delBuffer(base);
            base->resetDataPtr(mem);
        } else {
            engine.createBuffer(base, mem);
        }
    }

    // Handle the OpenCL context retrieval
    void* getDeviceContext() override {
        return engine.getCContext();
    };

    // Handle user kernels
    string userKernel(const std::string &kernel, std::vector<bh_view> &operand_list,
                      const std::string &compile_cmd, const std::string &tag) override {
        throw std::runtime_error("[OpenCL] userKernel not Implemented");
    }
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
        stat.write("OpenCL", config.defaultGet<std::string>("prof_filename", ""), cout);
    }
}

void Impl::execute(BhIR *bhir) {
    if (disabled) {
        child.execute(bhir);
        return;
    }

    if (to_col_major) {
        to_column_major(bhir->instr_list);
    }

    bh_base *cond = bhir->getRepeatCondition();
    for (uint64_t i = 0; i < bhir->getNRepeats(); ++i) {
        // Let's handle extension methods
        engine.handleExtmethod(bhir);

        // And then the regular instructions
        engine.handleExecution(bhir);

        // Check condition
        if (cond != nullptr) {
            engine.copyToHost({ cond }); // TODO: make it a read-only copy
            if (cond->getDataPtr() != nullptr and not((bool *) cond->getDataPtr())[0]) {
                break;
            }
        }
        // Change views that slide between iterations
        slide_views(bhir);
    }
}
