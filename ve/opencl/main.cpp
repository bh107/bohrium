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
#include <jitk/kernel.hpp>
#include <jitk/block.hpp>
#include <jitk/instruction.hpp>
#include <jitk/type.hpp>
#include <jitk/fuser.hpp>
#include <jitk/graph.hpp>
#include <jitk/transformer.hpp>

#include "engine_opencl.hpp"
#include "opencl_type.hpp"
#include "cl.hpp"

using namespace bohrium;
using namespace jitk;
using namespace component;
using namespace std;

namespace {
class Impl : public ComponentImplWithChild {
  private:
    // Known extension methods
    map<bh_opcode, extmethod::ExtmethodFace> extmethods;
    // Some statistics
    uint64_t num_base_arrays=0;
    uint64_t num_temp_arrays=0;
    uint64_t max_memory_usage=0;
    uint64_t totalwork=0;
    uint64_t threading_below_threshold=0;
    chrono::duration<double> time_total_execution{0};
    chrono::duration<double> time_fusion{0};
    chrono::duration<double> time_exec{0};
    chrono::duration<double> time_build{0};
    chrono::duration<double> time_offload{0};
    // The OpenCL engine
    EngineOpenCL engine;
    // Write an OpenCL kernel
    void write_kernel(const Kernel &kernel, BaseDB &base_ids, const vector<const LoopB *> &threaded_blocks,
                      stringstream &ss);

  public:
    Impl(int stack_level) : ComponentImplWithChild(stack_level), engine(config) {}
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

namespace {
void spaces(stringstream &out, int num) {
    for (int i = 0; i < num; ++i) {
        out << " ";
    }
}
}

Impl::~Impl() {
    if (config.defaultGet<bool>("prof", false)) {
        cout << "[OPENCL] Profiling: " << endl;
        cout << "\tArray Contractions:   " << num_temp_arrays << "/" << num_base_arrays << endl;
        cout << "\tMaximum Memory Usage: " << max_memory_usage / 1024 / 1024 << " MB" << endl;
        cout << "\tTotal Work: " << (double) totalwork << " operations" << endl;
        cout << "\tWork below par-threshold(1000): " \
             << threading_below_threshold / (double)totalwork * 100 << "%" << endl;
        cout << "\tTotal Execution:  " << time_total_execution.count() << "s" << endl;
        cout << "\t  Fusion:  " << time_fusion.count() << "s" << endl;
        cout << "\t  Build:   " << time_build.count() << "s" << endl;
        cout << "\t  Exec:    " << time_exec.count() << "s" << endl;
        cout << "\t  Offload: " << time_offload.count() << "s" << endl;
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

void write_loop_block(BaseDB &base_ids, const LoopB &block, const ConfigParser &config,
                      const vector<const LoopB *> &threaded_blocks, stringstream &out) {
    spaces(out, 4 + block.rank*4);

    if (block.isSystemOnly()) {
        out << "// Removed loop with only system instructions" << endl;
        return;
    }

    // Let's scalar replace reduction outputs that reduces over the innermost axis
    vector<bh_view> scalar_replacements;
    for (const InstrPtr instr: block._sweeps) {
        if (bh_opcode_is_reduction(instr->opcode) and sweeping_innermost_axis(instr)) {
            bh_base *base = instr->operand[0].base;
            if (base_ids.isTmp(base))
                continue; // No need to replace temporary arrays
            out << write_opencl_type(base->type) << " s" << base_ids[base] << ";" << endl;
            spaces(out, 4 + block.rank * 4);
            scalar_replacements.push_back(instr->operand[0]);
            base_ids.insertScalarReplacement(base);
        }
    }

    // We might not have to loop "peel" if all reduction have an identity value and writes to a scalar
    bool need_to_peel = false;
    {
        for (const InstrPtr instr: block._sweeps) {
            bh_base *b = instr->operand[0].base;
            if (not (has_reduce_identity(instr->opcode) and (base_ids.isScalarReplaced(b) or base_ids.isTmp(b)))) {
                need_to_peel = true;
                break;
            }
        }
    }

    // When not peeling, we need a neutral initial reduction value
    if (not need_to_peel) {
        for (const InstrPtr instr: block._sweeps) {
            bh_base *base = instr->operand[0].base;
            if (base_ids.isTmp(base))
                out << "t";
            else
                out << "s";
            out << base_ids[base] << " = ";
            write_reduce_identity(instr->opcode, base->type, out);
            out << "; // Neutral initial reduction value" << endl;
            spaces(out, 4 + block.rank * 4);
        }
    }

    // All local temporary arrays needs an variable declaration
    const set<bh_base*> local_tmps = block.getLocalTemps();

    // If this block is sweeped, we will "peel" the for-loop such that the
    // sweep instruction is replaced with BH_IDENTITY in the first iteration
    if (block._sweeps.size() > 0 and need_to_peel) {
        LoopB peeled_block(block);
        for (const InstrPtr instr: block._sweeps) {
            bh_instruction sweep_instr;
            sweep_instr.opcode = BH_IDENTITY;
            sweep_instr.operand[1] = instr->operand[1]; // The input is the same as in the sweep
            sweep_instr.operand[0] = instr->operand[0];
            // But the output needs an extra dimension when we are reducing to a non-scalar
            if (bh_opcode_is_reduction(instr->opcode) and instr->operand[1].ndim > 1) {
                sweep_instr.operand[0].insert_axis(instr->constant.get_int64(), 1, 0);
            }
            peeled_block.replaceInstr(instr, sweep_instr);
        }
        string itername;
        {stringstream t; t << "i" << block.rank; itername = t.str();}
        out << "{ // Peeled loop, 1. sweep iteration " << endl;
        spaces(out, 8 + block.rank*4);
        out << write_opencl_type(BH_UINT64) << " " << itername << " = 0;" << endl;
        // Write temporary array declarations
        for (bh_base* base: base_ids.getBases()) {
            if (local_tmps.find(base) != local_tmps.end()) {
                spaces(out, 8 + block.rank * 4);
                out << write_opencl_type(base->type) << " t" << base_ids[base] << ";" << endl;
            }
        }
        out << endl;
        for (const Block &b: peeled_block._block_list) {
            if (b.isInstr()) {
                if (b.getInstr() != NULL) {
                    spaces(out, 4 + b.rank()*4);
                    write_instr(base_ids, *b.getInstr(), out, true);
                }
            } else {
                write_loop_block(base_ids, b.getLoop(), config, threaded_blocks, out);
            }
        }
        spaces(out, 4 + block.rank*4);
        out << "}" << endl;
        spaces(out, 4 + block.rank*4);
    }

    // Write the for-loop header
    string itername;
    {stringstream t; t << "i" << block.rank; itername = t.str();}
    // Notice that we use find_if() with a lambda function since 'threaded_blocks' contains pointers not objects
    if (std::find_if(threaded_blocks.begin(), threaded_blocks.end(),
                     [&block](const LoopB* b){return *b == block;}) == threaded_blocks.end()) {
        out << "for(" << write_opencl_type(BH_UINT64) << " " << itername;
        if (block._sweeps.size() > 0 and need_to_peel) // If the for-loop has been peeled, we should start at 1
            out << "=1; ";
        else
            out << "=0; ";
        out << itername << " < " << block.size << "; ++" << itername << ") {" << endl;
    } else {
        assert(block._sweeps.size() == 0);
        out << "{ // Threaded block (ID " << itername << ")" << endl;
    }

    // Write temporary array declarations
    for (bh_base* base: base_ids.getBases()) {
        if (local_tmps.find(base) != local_tmps.end()) {
            spaces(out, 8 + block.rank * 4);
            out << write_opencl_type(base->type) << " t" << base_ids[base] << ";" << endl;
        }
    }

    // Write the for-loop body
    for (const Block &b: block._block_list) {
        if (b.isInstr()) { // Finally, let's write the instruction
            if (b.getInstr() != NULL) {
                spaces(out, 4 + b.rank()*4);
                write_instr(base_ids, *b.getInstr(), out, true);
            }
        } else {
            write_loop_block(base_ids, b.getLoop(), config, threaded_blocks, out);
        }
    }
    spaces(out, 4 + block.rank*4);
    out << "}" << endl;

    // Let's copy the scalar replacement back to the original array
    for (const bh_view &view: scalar_replacements) {
        spaces(out, 4 + block.rank*4);
        const size_t id = base_ids[view.base];
        out << "a" << id;
        write_array_subscription(view, out);
        out << " = s" << id << ";" << endl;
        base_ids.eraseScalarReplacement(view.base); // It is not scalar replaced anymore
    }
}

void Impl::write_kernel(const Kernel &kernel, BaseDB &base_ids, const vector<const LoopB *> &threaded_blocks,
                        stringstream &ss) {

    // Write the need includes
    ss << "#pragma OPENCL EXTENSION cl_khr_fp64: enable" << endl;
    ss << "#include <kernel_dependencies/complex_operations.h>" << endl;
    ss << "#include <kernel_dependencies/integer_operations.h>" << endl;
    if (kernel.useRandom()) { // Write the random function
        ss << "#include <kernel_dependencies/random123_opencl.h>" << endl;
    }
    ss << endl;

    // Write the header of the execute function
    ss << "__kernel void execute(";
    for(size_t i=0; i < kernel.getNonTemps().size(); ++i) {
        bh_base *b = kernel.getNonTemps()[i];
        ss << "__global " << write_opencl_type(b->type) << " a" << base_ids[b] << "[static " << b->nelem << "]";
        if (i+1 < kernel.getNonTemps().size()) {
            ss << ", ";
        }
    }
    ss << ") {" << endl;

    // Write the IDs of the threaded blocks
    if (threaded_blocks.size() > 0) {
        spaces(ss, 4);
        ss << "// The IDs of the threaded blocks: " << endl;
        for (unsigned int i=0; i < threaded_blocks.size(); ++i) {
            const LoopB *b = threaded_blocks[i];
            spaces(ss, 4);
            ss << "const " << write_opencl_type(BH_UINT64) << " i" << b->rank << " = get_global_id(" << i << "); " \
               << "if (i" << b->rank << " >= " << b->size << ") {return;} // Prevent overflow" << endl;
        }
        ss << endl;
    }

    // Write the block that makes up the body of 'execute()'
    write_loop_block(base_ids, kernel.block, config, threaded_blocks, ss);
    ss << "}" << endl << endl;

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
                if (v.base->data == NULL and not (util::exist(initiated, v.base) or util::exist(buffers, v.base))) {
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

    const bool verbose = config.defaultGet<bool>("verbose", false);

    // Let's start by cleanup the instructions from the 'bhir'
    vector<bh_instruction*> instr_list;
    {
        set<bh_base*> syncs;
        set<bh_base*> frees;
        instr_list = remove_non_computed_system_instr(bhir->instr_list, syncs, frees);

        // Let's copy sync'ed arrays back to the host
        engine.copyToHost(syncs);

        // Let's free device buffers
        for(bh_base *base: frees) {
            engine.buffers.erase(base);
        }
    }

    // Some statistics
    if (config.defaultGet<bool>("prof", false)) {
        for (const bh_instruction *instr: instr_list) {
            if (not bh_opcode_is_system(instr->opcode)) {
                totalwork += bh_nelements(instr->operand[0]);
            }
        }
    }
    auto tfusion = chrono::steady_clock::now();

    // Set the constructor flag
    set_constructor_flag(instr_list, engine.buffers);

    // Let's fuse the 'instr_list' into blocks
    vector<Block> block_list = fuser_singleton(instr_list);
    if (config.defaultGet<bool>("serial_fusion", false)) {
        fuser_serial(block_list, 1);
    } else {
    //    fuser_reshapable_first(block_list, 1);
        // Notice that the 'min_threading' argument is set to 1 in order to avoid blocks with no parallelism
        // TODO: Instead of always using 1, we could try once with no min threading before setting it to 1.
        fuser_greedy(block_list, 0);
        block_list = push_reductions_inwards(block_list);
        block_list = split_for_threading(block_list, 1000);
        block_list = collapse_redundant_axes(block_list);
    }

    // Pretty printing the block
    if (config.defaultGet<bool>("dump_graph", false)) {
        graph::DAG dag = graph::from_block_list(block_list);
        graph::pprint(dag, "dag");
    }

    time_fusion += chrono::steady_clock::now() - tfusion;

    for(const Block &block: block_list) {
        assert(not block.isInstr());

        //Let's create a kernel
        Kernel kernel(block.getLoop());

        // We can skip a lot of steps if the kernel does no computation
        const bool kernel_is_computing = not kernel.block.isSystemOnly();

        // For profiling statistic
        num_base_arrays += kernel.getNonTemps().size() + kernel.getAllTemps().size();
        num_temp_arrays += kernel.getAllTemps().size();

        // Assign IDs to all base arrays
        BaseDB base_ids;
        // NB: by assigning the IDs in the order they appear in the 'instr_list',
        //     the kernels can better be reused
        for (const InstrPtr instr: kernel.getAllInstr()) {
            const int nop = bh_noperands(instr->opcode);
            for(int i=0; i<nop; ++i) {
                const bh_view &v = instr->operand[i];
                if (not bh_is_constant(&v)) {
                    base_ids.insert(v.base);
                }
            }
        }
        base_ids.insertTmp(kernel.getAllTemps());

        // Debug print
        if (verbose) {
            cout << "Kernel's non-temps: " << endl;
            for (bh_base *base: kernel.getNonTemps()) {
                cout << "\t" << *base << endl;
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
                    threading_below_threshold += bh_nelements(instr->operand[0]);
                }
            }
        }

        // We might have to offload the execution to the CPU
        if (threaded_blocks.size() == 0 and kernel_is_computing) {
            if (verbose)
                cout << "Offloading to CPU" << endl;

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
            time_offload += chrono::steady_clock::now() - toffload;
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
                max_memory_usage = sum > max_memory_usage?sum:max_memory_usage;
            }

            // Code generation
            stringstream ss;
            write_kernel(kernel, base_ids, threaded_blocks, ss);

            if (verbose) {
                cout << endl << "************ GPU Kernel ************" << endl << ss.str()
                     << "^^^^^^^^^^^^ Kernel End ^^^^^^^^^^^^" << endl;
            }

            auto tkernel_exec = chrono::steady_clock::now();
            // Let's execute the OpenCL kernel
            engine.execute(ss.str(), kernel, threaded_blocks);
            time_exec += chrono::steady_clock::now() - tkernel_exec;
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
    time_total_execution += chrono::steady_clock::now() - texecution;
}

