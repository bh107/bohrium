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

    // The OpenCL context and device used throughout the execution
    cl::Context context;
    cl::Device default_device;
    // A map of allocated buffers on the device
    map<bh_base*, unique_ptr<cl::Buffer> > buffers;
    
    void write_kernel(const Kernel &kernel, BaseDB &base_ids, const vector<const Block*> &threaded_blocks, stringstream &ss);

  public:
    Impl(int stack_level) : ComponentImplWithChild(stack_level) {

        vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if(platforms.size() == 0) {
            throw runtime_error("No OpenCL platforms found");
        }
        cl::Platform default_platform=platforms[0];
        cout << "Using platform: " << default_platform.getInfo<CL_PLATFORM_NAME>() << endl;

        //get default device of the default platform
        vector<cl::Device> devices;
        default_platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
        if(devices.size()==0){
            throw runtime_error("No OpenCL device found");
        }
        default_device = devices[0];
        cout << "Using device: " << default_device.getInfo<CL_DEVICE_NAME>() << endl;

        vector<cl::Device> dev_list = {default_device};
        context = cl::Context(dev_list);
    }
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
        cout << "\t  Fusion: " << time_fusion.count() << "s" << endl;
        cout << "\t  Build:  " << time_build.count() << "s" << endl;
        cout << "\t  Exec:   " << time_exec.count() << "s" << endl;
    }
}

// Does 'instr' reduce over the innermost axis?
// Notice, that such a reduction computes each output element completely before moving
// to the next element.
bool sweeping_innermost_axis(const bh_instruction *instr) {
    if (not bh_opcode_is_sweep(instr->opcode))
        return false;
    assert(bh_noperands(instr->opcode) == 3);
    return sweep_axis(*instr) == instr->operand[1].ndim-1;
}

void write_block(BaseDB &base_ids, const Block &block, const ConfigParser &config,
                 const vector<const Block*> &threaded_blocks, stringstream &out) {
    assert(not block.isInstr());
    spaces(out, 4 + block.rank*4);

    if (block.isSystemOnly()) {
        out << "// Removed loop with only system instructions" << endl;
        return;
    }

    // Let's scalar replace reduction outputs that reduces over the innermost axis
    vector<bh_view> scalar_replacements;
    for (const bh_instruction *instr: block._sweeps) {
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
        for (const bh_instruction *instr: block._sweeps) {
            bh_base *b = instr->operand[0].base;
            if (not (has_reduce_identity(instr->opcode) and (base_ids.isScalarReplaced(b) or base_ids.isTmp(b)))) {
                need_to_peel = true;
                break;
            }
        }
    }

    // When not peeling, we need a neutral initial reduction value
    if (not need_to_peel) {
        for (const bh_instruction *instr: block._sweeps) {
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
        Block peeled_block(block);
        vector<bh_instruction> sweep_instr_list(block._sweeps.size());
        {
            size_t i = 0;
            for (const bh_instruction *instr: block._sweeps) {
                Block *sweep_instr_block = peeled_block.findInstrBlock(instr);
                assert(sweep_instr_block != NULL);
                bh_instruction *sweep_instr = &sweep_instr_list[i++];
                sweep_instr->opcode = BH_IDENTITY;
                sweep_instr->operand[1] = instr->operand[1]; // The input is the same as in the sweep
                sweep_instr->operand[0] = instr->operand[0];
                // But the output needs an extra dimension when we are reducing to a non-scalar
                if (bh_opcode_is_reduction(instr->opcode) and instr->operand[1].ndim > 1) {
                    sweep_instr->operand[0].insert_dim(instr->constant.get_int64(), 1, 0);
                }
                sweep_instr_block->_instr = sweep_instr;
            }
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
                if (b._instr != NULL) {
                    spaces(out, 4 + b.rank*4);
                    write_instr(base_ids, *b._instr, out, true);
                }
            } else {
                write_block(base_ids, b, config, threaded_blocks, out);
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
                     [&block](const Block* b){return *b == block;}) == threaded_blocks.end()) {
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
            if (b._instr != NULL) {
                spaces(out, 4 + b.rank*4);
                write_instr(base_ids, *b._instr, out, true);
            }
        } else {
            write_block(base_ids, b, config, threaded_blocks, out);
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

void Impl::write_kernel(const Kernel &kernel, BaseDB &base_ids, const vector<const Block*> &threaded_blocks, stringstream &ss) {

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
            const Block *b = threaded_blocks[i];
            spaces(ss, 4);
            ss << write_opencl_type(BH_UINT64) << " i" << b->rank << " = get_global_id(" << i << "); " \
               << "if (i" << b->rank << " >= " << b->size << ") {return;} // Prevent overflow" << endl;
        }
        ss << endl;
    }

    // Write the block that makes up the body of 'execute()'
    write_block(base_ids, kernel.block, config, threaded_blocks, ss);
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

// Returns the global and local work OpenCL ranges based on the 'threaded_blocks'
pair<cl::NDRange, cl::NDRange> NDRanges(const vector<const Block*> &threaded_blocks, const ConfigParser &config) {
    const auto &b = threaded_blocks;
    switch (b.size()) {
        case 1:
        {
            const cl_ulong lsize = config.defaultGet<cl_ulong>("work_group_size_1dx", 128);
            const cl_ulong rem = b[0]->size % lsize;
            const cl_ulong gsize = b[0]->size + (rem==0?0:(lsize-rem));
            return make_pair(cl::NDRange(gsize), cl::NDRange(lsize));
        }
        case 2:
        {
            const cl_ulong lsize_x = config.defaultGet<cl_ulong>("work_group_size_2dx", 32);
            const cl_ulong lsize_y = config.defaultGet<cl_ulong>("work_group_size_2dy", 4);
            const cl_ulong rem_x = b[0]->size % lsize_x;
            const cl_ulong rem_y = b[1]->size % lsize_y;
            const cl_ulong gsize_x = b[0]->size + (rem_x==0?0:(lsize_x-rem_x));
            const cl_ulong gsize_y = b[1]->size + (rem_y==0?0:(lsize_y-rem_y));
            return make_pair(cl::NDRange(gsize_x, gsize_y), cl::NDRange(lsize_x, lsize_y));
        }
        case 3:
        {
            const cl_ulong lsize_x = config.defaultGet<cl_ulong>("work_group_size_3dx", 32);
            const cl_ulong lsize_y = config.defaultGet<cl_ulong>("work_group_size_3dy", 2);
            const cl_ulong lsize_z = config.defaultGet<cl_ulong>("work_group_size_3dz", 2);
            const cl_ulong rem_x = b[0]->size % lsize_x;
            const cl_ulong rem_y = b[1]->size % lsize_y;
            const cl_ulong rem_z = b[2]->size % lsize_z;
            const cl_ulong gsize_x = b[0]->size + (rem_x==0?0:(lsize_x-rem_x));
            const cl_ulong gsize_y = b[1]->size + (rem_y==0?0:(lsize_y-rem_y));
            const cl_ulong gsize_z = b[2]->size + (rem_z==0?0:(lsize_z-rem_z));
            return make_pair(cl::NDRange(gsize_x, gsize_y, gsize_z), cl::NDRange(lsize_x, lsize_y, lsize_z));
        }
        default:
            throw runtime_error("NDRanges: maximum of three dimensions!");
    }
}

void Impl::execute(bh_ir *bhir) {
    auto texecution = chrono::steady_clock::now();

    const bool verbose = config.defaultGet<bool>("verbose", false);

    cl::CommandQueue queue(context, default_device);

    // Let's start by cleanup the instructions from the 'bhir'
    vector<bh_instruction*> instr_list;
    {
        set<bh_base*> syncs;
        set<bh_base*> frees;
        instr_list = remove_non_computed_system_instr(bhir->instr_list, syncs, frees);

        // Let's copy sync'ed arrays back to the host
        for(bh_base *base: syncs) {
            if (buffers.find(base) != buffers.end()) {
                bh_data_malloc(base);
                if (verbose) {
                    cout << "Copy to host: " << *base << endl;
                }
                queue.enqueueReadBuffer(*buffers.at(base), CL_FALSE, 0, (cl_ulong) bh_base_size(base), base->data);
                // When syncing we assume that the host writes to the data and invalidate the device data thus
                // we have to remove its data buffer
                buffers.erase(base);
            }
        }
        queue.finish();

        // Let's free device buffers
        for(bh_base *base: frees) {
            buffers.erase(base);
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
    set_constructor_flag(instr_list, buffers);

    // Let's fuse the 'instr_list' into blocks
    vector<Block> block_list = fuser_singleton(instr_list);
    if (config.defaultGet<bool>("serial_fusion", false)) {
        fuser_serial(block_list);
    } else {
    //  fuser_reshapable_first(block_list);
        fuser_greedy(block_list);
    }

    // Pretty printing the block
    if (config.defaultGet<bool>("dump_graph", false)) {
        graph::DAG dag = graph::from_block_list(block_list);
        graph::pprint(dag, "dag");
    }

    time_fusion += chrono::steady_clock::now() - tfusion;

    for(const Block &block: block_list) {

        //Let's create a kernel
        Kernel kernel(block);

        // We can skip a lot of steps if the kernel does no computation
        const bool kernel_is_computing = not kernel.block.isSystemOnly();

        // For profiling statistic
        num_base_arrays += kernel.getNonTemps().size() + kernel.getAllTemps().size();
        num_temp_arrays += kernel.getAllTemps().size();

        // Assign IDs to all base arrays
        BaseDB base_ids;
        // NB: by assigning the IDs in the order they appear in the 'instr_list',
        //     the kernels can better be reused
        for (const bh_instruction *instr: kernel.getAllInstr()) {
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
        vector<const Block*> threaded_blocks;
        uint64_t total_threading;
        tie(threaded_blocks, total_threading) = find_threaded_blocks(kernel.block);
        if (total_threading < config.defaultGet<uint64_t>("parallel_threshold", 1000)) {
            for (const bh_instruction *instr: kernel.getAllInstr()) {
                if (not bh_opcode_is_system(instr->opcode)) {
                    threading_below_threshold += bh_nelements(instr->operand[0]);
                }
            }
        }

        // We might have to offload the execution to the CPU
        if (threaded_blocks.size() == 0 and kernel_is_computing) {
            if (verbose)
                cout << "Offloading to CPU" << endl;

            // Let's copy all non-temporary to the host
            for (bh_base *base: kernel.getNonTemps()) {
                if (buffers.find(base) != buffers.end()) {
                    bh_data_malloc(base);
                    if (verbose) {
                        cout << "Copy to host: " << *base << endl;
                    }
                    queue.enqueueReadBuffer(*buffers.at(base), CL_TRUE, 0, (cl_ulong) bh_base_size(base), base->data);
                }
            }
            queue.finish();

            // Let's free device buffers
            for (bh_base *base: kernel.getFrees()) {
                buffers.erase(base);
            }
            for (bh_base *base: kernel.getNonTemps()) {
                buffers.erase(base);
            }

            // Let's send the kernel instructions to our child
            vector<bh_instruction> child_instr_list;
            for (const bh_instruction* instr: kernel.block.getAllInstr()) {
                child_instr_list.push_back(*instr);
            }
            bh_ir tmp_bhir(child_instr_list.size(), &child_instr_list[0]);
            child.execute(&tmp_bhir);
            continue;
        }

        // Let's execute the kernel
        if (kernel_is_computing) {

            // Code generation
            stringstream ss;
            write_kernel(kernel, base_ids, threaded_blocks, ss);

            if (verbose) {
                cout << endl << "************ GPU Kernel ************" << endl << ss.str()
                             << "^^^^^^^^^^^^ Kernel End ^^^^^^^^^^^^" << endl;
            }

            auto tkernel_build = chrono::steady_clock::now();
            cl::Program program(context, ss.str());
            const string compile_inc = config.defaultGet<string>("compiler_flg", "");
            try {
                program.build({default_device}, compile_inc.c_str());
                if (verbose) {
                    cout << "************ Build Log ************" << endl \
                         << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) \
                         << "^^^^^^^^^^^^^ Log END ^^^^^^^^^^^^^" << endl << endl;
                }
            } catch (cl::Error e) {
                cerr << "Error building: " << endl << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << endl;
                throw;
            }
            time_build += chrono::steady_clock::now() - tkernel_build;

            // We need a memory buffer on the device for each non-temporary array in the kernel
            for(bh_base *base: kernel.getNonTemps()) {
                if (buffers.find(base) == buffers.end()) { // We shouldn't overwrite existing buffers
                    cl::Buffer *b = new cl::Buffer(context, CL_MEM_READ_WRITE, (cl_ulong) bh_base_size(base));
                    buffers[base].reset(b);

                    // If the host data is non-null we should copy it to the device
                    if (base->data != NULL) {
                        if (verbose) {
                            cout << "Copy to device: " << *base << endl;
                        }
                        queue.enqueueWriteBuffer(*b, CL_FALSE, 0, (cl_ulong) bh_base_size(base), base->data);
                    }
                }
            }
            queue.finish();

            if (config.defaultGet<bool>("prof", false)) {
                // Let's check the current memory usage on the device
                uint64_t sum = 0;
                for (const auto &b: buffers) {
                    sum += bh_base_size(b.first);
                }
                max_memory_usage = sum > max_memory_usage?sum:max_memory_usage;
            }

            auto tkernel_exec = chrono::steady_clock::now();
            // Let's execute the OpenCL kernel
            cl::Kernel opencl_kernel = cl::Kernel(program, "execute");
            {
                cl_uint i = 0;
                for (bh_base *base: kernel.getNonTemps()) { // NB: the iteration order matters!
                    opencl_kernel.setArg(i++, *buffers.at(base));
                }
            }
            const auto ranges = NDRanges(threaded_blocks, config);
            queue.enqueueNDRangeKernel(opencl_kernel, cl::NullRange, ranges.first, ranges.second);
            queue.finish();
            time_exec += chrono::steady_clock::now() - tkernel_exec;
        }

        // Let's copy sync'ed arrays back to the host
        for(bh_base *base: kernel.getSyncs()) {
            if (buffers.find(base) != buffers.end()) {
                bh_data_malloc(base);
                if (verbose) {
                    cout << "Copy to host: " << *base << endl;
                }
                queue.enqueueReadBuffer(*buffers.at(base), CL_FALSE, 0, (cl_ulong) bh_base_size(base), base->data);
                // When syncing we assume that the host writes to the data and invalidate the device data thus
                // we have to remove its data buffer
                buffers.erase(base);
            }
        }
        queue.finish();

        // Let's free device buffers
        const auto &kernel_frees = kernel.getFrees();
        for(bh_base *base: kernel.getFrees()) {
            buffers.erase(base);
        }

        // Finally, let's cleanup
        for(bh_base *base: kernel_frees) {
            bh_data_free(base);
        }
    }
    time_total_execution += chrono::steady_clock::now() - texecution;
}

