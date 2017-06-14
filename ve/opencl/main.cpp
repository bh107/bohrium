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
#include <bh_opcode.h>
#include <jitk/statistics.hpp>
#include <jitk/kernel.hpp>
#include <jitk/block.hpp>
#include <jitk/instruction.hpp>
#include <jitk/fuser.hpp>
#include <jitk/graph.hpp>
#include <jitk/transformer.hpp>
#include <jitk/fuser_cache.hpp>
#include <jitk/codegen_util.hpp>
#include <jitk/dtype.hpp>
#include <jitk/apply_fusion.hpp>

#include "engine_opencl.hpp"

using namespace bohrium;
using namespace jitk;
using namespace component;
using namespace std;

namespace {
class Impl : public ComponentImplWithChild {
  private:
    // Some statistics
    Statistics stat;
    // Fuse cache
    FuseCache fcache;
    // Known extension methods
    map<bh_opcode, extmethod::ExtmethodFace> extmethods;
    set<bh_opcode> child_extmethods;
    // The OpenCL engine
    EngineOpenCL engine;

public:
    Impl(int stack_level) : ComponentImplWithChild(stack_level), stat(config.defaultGet("prof", false)),
                            fcache(stat), engine(config, stat) {}
    ~Impl();
    void execute(bh_ir *bhir);
    void extmethod(const string &name, bh_opcode opcode) {
        // ExtmethodFace does not have a default or copy constructor thus
        // we have to use its move constructor.
        try {
            extmethods.insert(make_pair(opcode, extmethod::ExtmethodFace(config, name)));
        } catch(extmethod::ExtmethodNotFound e) {
            // I don't know this function, lets try my child
            child.extmethod(name, opcode);
            child_extmethods.insert(opcode);
        }
    }

    // Write an OpenCL kernel
    void write_kernel(const Kernel &kernel, const SymbolTable &symbols, const ConfigParser &config,
                      const vector<const LoopB *> &threaded_blocks,
                      const vector<const bh_view*> &offset_strides, stringstream &ss);

    // Implement the handle of extension methods
    void handle_extmethod(bh_ir *bhir) {
        util_handle_extmethod(this, bhir, extmethods, child_extmethods, child, &engine);
    }

    // Returns the blocks that can be parallelized in 'kernel' (incl. sub-blocks)
    vector<const LoopB*> find_threaded_blocks(Kernel &kernel) {
        vector<const LoopB*> threaded_blocks;
        uint64_t total_threading;
        tie(threaded_blocks, total_threading) = util_find_threaded_blocks(kernel.block);
        if (total_threading < config.defaultGet<uint64_t>("parallel_threshold", 1000)) {
            for (const InstrPtr instr: kernel.getAllInstr()) {
                if (not bh_opcode_is_system(instr->opcode)) {
                    stat.threading_below_threshold += bh_nelements(instr->operand[0]);
                }
            }
        }
        return threaded_blocks;
    }

    // Handle messages from parent
    string message(const string &msg) {
        stringstream ss;
        if (msg == "statistic_enable_and_reset") {
            stat = Statistics(true, config.defaultGet("prof", false));
        } else if (msg == "statistic") {
            stat.write("OpenCL", "", ss);
        } else if (msg == "GPU: disable") {
            engine.allBasesToHost();
            disabled = true;
        } else if (msg == "GPU: enable") {
            disabled = false;
        }
        return ss.str() + child.message(msg);
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
        stat.write("OpenCL", config.defaultGet<std::string>("prof_filename", ""), cout);
    }
}


// Writes the OpenCL specific for-loop header
void loop_head_writer(const SymbolTable &symbols, Scope &scope, const LoopB &block, const ConfigParser &config,
                      bool loop_is_peeled, const vector<const LoopB *> &threaded_blocks, stringstream &out) {
    // Write the for-loop header
    string itername;
    {stringstream t; t << "i" << block.rank; itername = t.str();}
    // Notice that we use find_if() with a lambda function since 'threaded_blocks' contains pointers not objects
    if (std::find_if(threaded_blocks.begin(), threaded_blocks.end(),
                     [&block](const LoopB* b){return *b == block;}) == threaded_blocks.end()) {
        out << "for(" << write_opencl_type(bh_type::UINT64) << " " << itername;
        if (block._sweeps.size() > 0 and loop_is_peeled) // If the for-loop has been peeled, we should start at 1
            out << "=1; ";
        else
            out << "=0; ";
        out << itername << " < " << block.size << "; ++" << itername << ") {\n";
    } else {
        assert(block._sweeps.size() == 0);
        out << "{ // Threaded block (ID " << itername << ")\n";
    }
}

void Impl::write_kernel(const Kernel &kernel, const SymbolTable &symbols, const ConfigParser &config,
                        const vector<const LoopB *> &threaded_blocks,
                        const vector<const bh_view*> &offset_strides, stringstream &ss) {

    // Write the need includes
    ss << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
    ss << "#include <kernel_dependencies/complex_opencl.h>\n";
    ss << "#include <kernel_dependencies/integer_operations.h>\n";
    if (kernel.useRandom()) { // Write the random function
        ss << "#include <kernel_dependencies/random123_opencl.h>\n";
    }
    ss << "\n";

    // Write the header of the execute function
    ss << "__kernel void execute";
    write_kernel_function_arguments(kernel, symbols, offset_strides, write_opencl_type, ss, "__global", false);
    ss << "{\n";

    // Write the IDs of the threaded blocks
    if (threaded_blocks.size() > 0) {
        spaces(ss, 4);
        ss << "// The IDs of the threaded blocks: \n";
        for (unsigned int i=0; i < threaded_blocks.size(); ++i) {
            const LoopB *b = threaded_blocks[i];
            spaces(ss, 4);
            ss << "const " << write_opencl_type(bh_type::UINT32) << " i" << b->rank << " = get_global_id(" << i << "); " \
               << "if (i" << b->rank << " >= " << b->size << ") {return;} // Prevent overflow\n";
        }
        ss << "\n";
    }

    // Write the block that makes up the body of 'execute()'
    write_loop_block(symbols, NULL, kernel.block, config, threaded_blocks, true, write_opencl_type, loop_head_writer, ss);

    ss << "}\n\n";
}


void Impl::execute(bh_ir *bhir) {
    if (disabled) {
        child.execute(bhir);
        return;
    }

    // Let's handle extension methods
    util_handle_extmethod(this, bhir, extmethods, child_extmethods, child, &engine);

    // And then the regular instructions
    handle_execution(*this, bhir, engine, config, stat, fcache, &child);
}
