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
    // Write an OpenCL kernel
    void write_kernel(const Kernel &kernel, const SymbolTable &symbols, const vector<const LoopB *> &threaded_blocks,
                      const vector<const bh_view*> &offset_strides, stringstream &ss);

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
};
}

extern "C" ComponentImpl* create(int stack_level) {
    return new Impl(stack_level);
}
extern "C" void destroy(ComponentImpl* self) {
    delete self;
}

Impl::~Impl() {
    if (config.defaultGet<bool>("prof", false)) {
        stat.pprint("OpenCL", cout);
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
        out << "for(" << write_opencl_type(BH_UINT64) << " " << itername;
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

void Impl::write_kernel(const Kernel &kernel, const SymbolTable &symbols, const vector<const LoopB *> &threaded_blocks,
                        const vector<const bh_view*> &offset_strides, stringstream &ss) {

    // Write the need includes
    ss << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
    ss << "#include <kernel_dependencies/complex_operations.h>\n";
    ss << "#include <kernel_dependencies/integer_operations.h>\n";
    if (kernel.useRandom()) { // Write the random function
        ss << "#include <kernel_dependencies/random123_opencl.h>\n";
    }
    ss << "\n";

    // Write the header of the execute function
    ss << "__kernel void execute";
    write_kernel_function_arguments(kernel, symbols, offset_strides, write_opencl_type, ss, "__global");
    ss << "{\n";

    // Write the IDs of the threaded blocks
    if (threaded_blocks.size() > 0) {
        spaces(ss, 4);
        ss << "// The IDs of the threaded blocks: \n";
        for (unsigned int i=0; i < threaded_blocks.size(); ++i) {
            const LoopB *b = threaded_blocks[i];
            spaces(ss, 4);
            ss << "const " << write_opencl_type(BH_UINT32) << " i" << b->rank << " = get_global_id(" << i << "); " \
               << "if (i" << b->rank << " >= " << b->size << ") {return;} // Prevent overflow\n";
        }
        ss << "\n";
    }

    // Write the block that makes up the body of 'execute()'
    write_loop_block(symbols, NULL, kernel.block, config, threaded_blocks, true, write_opencl_type, loop_head_writer, ss);

    ss << "}\n\n";
}


void Impl::execute(bh_ir *bhir) {
    auto texecution = chrono::steady_clock::now();

    const bool verbose = config.defaultGet<bool>("verbose", false);
    const bool strides_as_variables = config.defaultGet<bool>("strides_as_variables", true);

    // Let's handle extension methods
    util_handle_extmethod(this, bhir, extmethods, child_extmethods, child, &engine);

    // Some statistics
    stat.record(bhir->instr_list);

    // Let's start by cleanup the instructions from the 'bhir'
    vector<bh_instruction*> instr_list;
    {
        set<bh_base*> syncs;
        set<bh_base*> frees;
        instr_list = remove_non_computed_system_instr(bhir->instr_list, syncs, frees);

        // Let's copy sync'ed arrays back to the host
        engine.copyToHost(syncs);

        // Let's free device buffers and array memory
        for(bh_base *base: frees) {
            engine.buffers.erase(base);
            bh_data_free(base);
        }
    }

    // Set the constructor flag
    if (config.defaultGet<bool>("array_contraction", true)) {
        util_set_constructor_flag(instr_list, engine.buffers);
    } else {
        for (bh_instruction *instr: instr_list) {
            instr->constructor = false;
        }
    }

    // The cache system
    const vector<Block> block_list = get_block_list(instr_list, config, fcache, stat);

    for(const Block &block: block_list) {
        assert(not block.isInstr());

        //Let's create a kernel
        Kernel kernel = create_kernel_object(block, verbose, stat);

        const SymbolTable symbols(kernel.getAllInstr(),
                                  config.defaultGet("index_as_var", true),
                                  config.defaultGet("const_as_var", true));

        // We can skip a lot of steps if the kernel does no computation
        const bool kernel_is_computing = not kernel.block.isSystemOnly();

        // Find the parallel blocks
        vector<const LoopB*> threaded_blocks;
        uint64_t total_threading;
        tie(threaded_blocks, total_threading) = find_threaded_blocks(kernel.block);
        if (total_threading < config.defaultGet<uint64_t>("parallel_threshold", 1000)) {
            for (const InstrPtr instr: kernel.getAllInstr()) {
                if (not bh_opcode_is_system(instr->opcode)) {
                    stat.threading_below_threshold += bh_nelements(instr->operand[0]);
                }
            }
        }

        // We might have to offload the execution to the CPU
        if (threaded_blocks.size() == 0 and kernel_is_computing) {
            if (verbose)
                cout << "Offloading to CPU\n";

            auto toffload = chrono::steady_clock::now();

            // Let's copy all non-temporary to the host
            engine.copyToHost(kernel.getNonTemps());

            // Let's free device buffers
            for (bh_base *base: kernel.getFrees()) {
                engine.buffers.erase(base);
            }

            // Let's send the kernel instructions to our child
            vector<bh_instruction> child_instr_list;
            for (const InstrPtr instr: kernel.block.getAllInstr()) {
                child_instr_list.push_back(*instr);
            }
            bh_ir tmp_bhir(child_instr_list.size(), &child_instr_list[0]);
            child.execute(&tmp_bhir);
            stat.time_offload += chrono::steady_clock::now() - toffload;
            continue;
        }

        // Let's execute the kernel
        if (kernel_is_computing) {

            // We need a memory buffer on the device for each non-temporary array in the kernel
            engine.copyToDevice(kernel.getNonTemps());

            if (config.defaultGet<bool>("prof", false)) {
                // Let's check the current memory usage on the device
                uint64_t sum = 0;
                for (const auto &b: engine.buffers) {
                    sum += bh_base_size(b.first);
                }
                stat.max_memory_usage = sum > stat.max_memory_usage?sum:stat.max_memory_usage;
            }

            // Get the offset and strides (an empty 'offset_strides' deactivate "strides as variables")
            vector<const bh_view*> offset_strides;
            if (strides_as_variables) {
                offset_strides = kernel.getOffsetAndStrides();
            }

            // Code generation
            stringstream ss;
            write_kernel(kernel, symbols, threaded_blocks, offset_strides, ss);

            if (verbose) {
                cout << "\n************ GPU Kernel ************\n" << ss.str()
                     << "^^^^^^^^^^^^ Kernel End ^^^^^^^^^^^^" << endl;
            }

            // Create the constant vector
            vector<const bh_instruction*> constants;
            constants.reserve(symbols.constIDs().size());
            for (const InstrPtr &instr: symbols.constIDs()) {
                constants.push_back(&(*instr));
            }

            // Let's execute the OpenCL kernel
            engine.execute(ss.str(), kernel, threaded_blocks, offset_strides, constants);
        }

        // Let's copy sync'ed arrays back to the host
        engine.copyToHost(kernel.getSyncs());

        // Let's free device buffers
        const auto &kernel_frees = kernel.getFrees();
        for(bh_base *base: kernel.getFrees()) {
            engine.buffers.erase(base);
        }

        // Finally, let's cleanup
        for(bh_base *base: kernel_frees) {
            bh_data_free(base);
        }
    }
    stat.time_total_execution += chrono::steady_clock::now() - texecution;
}
