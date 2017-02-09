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

#include "engine_opencl.hpp"
#include "opencl_type.hpp"

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
        extmethods.insert(make_pair(opcode, extmethod::ExtmethodFace(config, name)));
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

// Does 'instr' reduce over the innermost axis?
// Notice, that such a reduction computes each output element completely before moving
// to the next element.
bool sweeping_innermost_axis(InstrPtr instr) {
    if (not bh_opcode_is_sweep(instr->opcode))
        return false;
    assert(bh_noperands(instr->opcode) == 3);
    return instr->sweep_axis() == instr->operand[1].ndim-1;
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
    ss << "__kernel void execute(";
    for(size_t i=0; i < kernel.getNonTemps().size(); ++i) {
        bh_base *b = kernel.getNonTemps()[i];
        ss << "__global " << write_opencl_type(b->type) << " *a" << symbols.baseID(b);
        if (i+1 < kernel.getNonTemps().size()) {
            ss << ", ";
        }
    }
    // Let's find all offset-and-strides of all non-temporary arrays in the order the appear in the instruction list
    for (const bh_view *view: offset_strides) {
        ss << ", " << write_opencl_type(BH_UINT64) << " vo" << symbols.offsetStridesID(*view);
        for (int i=0; i<view->ndim; ++i) {
            ss << ", " << write_opencl_type(BH_UINT64) << " vs" << symbols.offsetStridesID(*view) << "_" << i;
        }
    }
    ss << ") {\n";

    // Write the IDs of the threaded blocks
    if (threaded_blocks.size() > 0) {
        spaces(ss, 4);
        ss << "// The IDs of the threaded blocks: \n";
        for (unsigned int i=0; i < threaded_blocks.size(); ++i) {
            const LoopB *b = threaded_blocks[i];
            spaces(ss, 4);
            ss << "const " << write_opencl_type(BH_UINT64) << " i" << b->rank << " = get_global_id(" << i << "); " \
               << "if (i" << b->rank << " >= " << b->size << ") {return;} // Prevent overflow\n";
        }
        ss << "\n";
    }

    // Write the block that makes up the body of 'execute()'
    write_loop_block(symbols, NULL, kernel.block, config, threaded_blocks, true, write_opencl_type, loop_head_writer, ss);

    ss << "}\n\n";
}

// Sets the constructor flag of each instruction in 'instr_list'
void set_constructor_flag(vector<bh_instruction*> &instr_list, const map<bh_base*, unique_ptr<cl::Buffer> > &buffers) {
    set<bh_base*> initiated; // Arrays initiated in 'instr_list'
    for(bh_instruction *instr: instr_list) {
        instr->constructor = false;
        int nop = bh_noperands(instr->opcode);
        for (bh_intp o = 0; o < nop; ++o) {
            const bh_view &v = instr->operand[o];
            if (not bh_is_constant(&v)) {
                assert(v.base != NULL);
                if (v.base->data == NULL and not (util::exist_nconst(initiated, v.base)
                                                  or util::exist_nconst(buffers, v.base))) {
                    if (o == 0) { // It is only the output that is initiated
                        initiated.insert(v.base);
                        instr->constructor = true;
                    }
                }
            }
        }
    }
}

void Impl::execute(bh_ir *bhir) {
    auto texecution = chrono::steady_clock::now();

    // Some statistics
    stat.record(bhir->instr_list);

    // For now, we handle extension methods by executing them individually
    {
        vector<bh_instruction> instr_list;
        for (bh_instruction &instr: bhir->instr_list) {
            auto ext = extmethods.find(instr.opcode);
            if (ext != extmethods.end()) {
                // Execute the instructions up until now
                bh_ir b;
                b.instr_list = instr_list;
                this->execute(&b);
                instr_list.clear();

                // Execute the extension method
                ext->second.execute(&instr, &engine);
            } else {
                instr_list.push_back(instr);
            }
        }
        bhir->instr_list = instr_list;
    }

    const bool verbose = config.defaultGet<bool>("verbose", false);
    const bool strides_as_variables = config.defaultGet<bool>("strides_as_variables", true);

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
        set_constructor_flag(instr_list, engine.buffers);
    }

    // The cache system
    vector<Block> block_list;
    {
        // Assign origin ids to all instructions starting at zero.
        int64_t count = 0;
        for (bh_instruction *instr: instr_list) {
            instr->origin_id = count++;
        }

        bool hit;
        tie(block_list, hit) = fcache.get(instr_list);
        if (hit) {
;
        } else {
            const auto tpre_fusion = chrono::steady_clock::now();
            stat.num_instrs_into_fuser += instr_list.size();
            // Let's fuse the 'instr_list' into blocks
            // We start with the pre_fuser
            if (config.defaultGet<bool>("pre_fuser", true)) {
                block_list = pre_fuser_lossy(instr_list);
            } else {
                block_list = fuser_singleton(instr_list);
            }
            stat.num_blocks_out_of_fuser += block_list.size();
            const auto tfusion = chrono::steady_clock::now();
            stat.time_pre_fusion +=  tfusion - tpre_fusion;
            // Then we fuse fully
            if (config.defaultGet<bool>("serial_fusion", false)) {
                fuser_serial(block_list, 1);
            } else {
                fuser_greedy(block_list, 0);
                block_list = push_reductions_inwards(block_list);
                block_list = split_for_threading(block_list, 1000);
                block_list = collapse_redundant_axes(block_list);
            }
            stat.time_fusion += chrono::steady_clock::now() - tfusion;
            fcache.insert(instr_list, block_list);
        }
    }

    // Pretty printing the block
    if (config.defaultGet<bool>("graph", false)) {
        graph::DAG dag = graph::from_block_list(block_list);
        graph::pprint(dag, "dag");
    }

    for(const Block &block: block_list) {
        assert(not block.isInstr());

        //Let's create a kernel
        Kernel kernel(block.getLoop());

        // We can skip a lot of steps if the kernel does no computation
        const bool kernel_is_computing = not kernel.block.isSystemOnly();

        // For profiling statistic
        stat.num_base_arrays += kernel.getNonTemps().size() + kernel.getAllTemps().size();
        stat.num_temp_arrays += kernel.getAllTemps().size();

        const SymbolTable symbols(kernel.getAllInstr());

        // Debug print
        if (verbose) {
            cout << "Kernel's non-temps: \n";
            for (bh_base *base: kernel.getNonTemps()) {
                cout << "\t" << *base << "\n";
            }
            cout << kernel.block;
        }

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
            engine.copyListToDevice(kernel.getNonTemps());

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

            // Let's execute the OpenCL kernel
            engine.execute(ss.str(), kernel, threaded_blocks, offset_strides);
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
