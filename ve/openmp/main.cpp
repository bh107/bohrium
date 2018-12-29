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
                      const std::string &compile_cmd, const std::string &tag) override;
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


string Impl::userKernel(const std::string &kernel, std::vector<bh_view> &operand_list,
                        const std::string &compile_cmd, const std::string &tag) {

    for (const bh_view &op: operand_list) {
        if (op.isConstant()) {
            return "[UserKernel] fatal error - operands cannot be constants";
        }
        bh_data_malloc(op.base);
    }
    string kernel_with_launcher;
    vector<void *> data_list;
    {
        stringstream ss;
        ss << kernel << "\n";
        ss << "void _bh_launcher(void *data_list[]) {\n";
        for (size_t i=0; i<operand_list.size(); ++i) {
            ss << "    " << engine.writeType(operand_list[i].base->dtype());
            ss << " *a" << i << " = data_list[" << i << "];\n";
            data_list.push_back(operand_list[i].base->getDataPtr());
        }
        ss << "    execute(";
        for (size_t i=0; i<operand_list.size()-1; ++i) {
            ss << "a" << i << ", ";
        }
        if (not operand_list.empty()) {
            ss << "a" << operand_list.size()-1;
        }
        ss << ");\n";
        ss << "}\n";
        kernel_with_launcher = ss.str();
    }

    UserKernelFunction func;
    try {
        KernelFunction f = engine.getFunction(kernel_with_launcher, "_bh_launcher", compile_cmd);
        func = reinterpret_cast<UserKernelFunction>(f);
        assert(func != nullptr);
    } catch (const std::runtime_error &e) {
        return string(e.what());
    }

    func(&data_list[0]);
    return "";
}
