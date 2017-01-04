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
#include <jitk/kernel.hpp>
#include <jitk/block.hpp>
#include <jitk/instruction.hpp>
#include <jitk/graph.hpp>
#include <jitk/transformer.hpp>
#include <jitk/fuser_cache.hpp>
#include <jitk/codegen_util.hpp>

#include "store.hpp"
#include "c99_type.hpp"
#include "openmp_util.hpp"

using namespace bohrium;
using namespace jitk;
using namespace component;
using namespace std;

namespace {
class Impl : public ComponentImpl {
  private:
    // Fuse cache
    FuseCache fcache;
    // Compiled kernels store
    Store _store;
    // Known extension methods
    map<bh_opcode, extmethod::ExtmethodFace> extmethods;
    //Allocated base arrays
    set<bh_base*> _allocated_bases;
    // Some statistics
    uint64_t num_base_arrays=0;
    uint64_t num_temp_arrays=0;
    uint64_t totalwork=0;
    chrono::duration<double> time_total_execution{0};
    chrono::duration<double> time_fusion{0};
    chrono::duration<double> time_exec{0};
    chrono::duration<double> time_build{0};
  public:
    Impl(int stack_level) : ComponentImpl(stack_level), _store(config) {}
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
        const int64_t store_hits = _store.num_lookups - _store.num_lookup_misses;
        const uint64_t fcache_hits = fcache.num_lookups - fcache.num_lookup_misses;
        cout << "[VE-OPENMP] Profiling: \n";
        cout << "\tKernel Store Hits:   " << store_hits << "/" << _store.num_lookups \
                                          << " (" << 100.0*store_hits/_store.num_lookups << "%)\n";
        cout << "\tFuse Cache hits:     " << fcache_hits << "/" << fcache.num_lookups \
                                          << " (" << 100.0*fcache_hits/fcache.num_lookups << "%)\n";
        cout << "\tArray contractions:  " << num_temp_arrays << "/" << num_base_arrays \
                                          << " (" << 100.0*num_temp_arrays/num_base_arrays << "%)\n";
        cout << "\tTotal Work: " << (double) totalwork << " operations\n";
        cout << "\tTotal Execution:  " << time_total_execution.count() << "s\n";
        cout << "\t  Fusion: " << time_fusion.count() << "s\n";
        cout << "\t  Build:  " << time_build.count() << "s\n";
        cout << "\t  Exec:   " << time_exec.count() << "s" << endl;
    }
}

// Writing the OpenMP header, which include "parallel for" and "simd"
void write_openmp_header(const LoopB &block, BaseDB &base_ids, const ConfigParser &config, stringstream &out) {
    if (not config.defaultGet<bool>("compiler_openmp", false)) {
        return;
    }
    bool enable_simd = config.defaultGet<bool>("compiler_openmp_simd", false);

    // All reductions that can be handle directly be the OpenMP header e.g. reduction(+:var)
    vector<InstrPtr> openmp_reductions;

    stringstream ss;
    // "OpenMP for" goes to the outermost loop
    if (block.rank == 0 and openmp_compatible(block)) {
        ss << " parallel for";
        // Since we are doing parallel for, we should either do OpenMP reductions or protect the sweep instructions
        for (const InstrPtr instr: block._sweeps) {
            assert(bh_noperands(instr->opcode) == 3);
            bh_base *base = instr->operand[0].base;
            if (openmp_reduce_compatible(instr->opcode) and (base_ids.isScalarReplaced(base) or base_ids.isTmp(base))) {
                openmp_reductions.push_back(instr);
            } else if (openmp_atomic_compatible(instr->opcode)) {
                base_ids.insertOpenmpAtomic(instr->operand[0].base);
            } else {
                base_ids.insertOpenmpCritical(instr->operand[0].base);
            }
        }
    }

    // "OpenMP SIMD" goes to the innermost loop (which might also be the outermost loop)
    if (enable_simd and block.isInnermost() and simd_compatible(block, base_ids)) {
        ss << " simd";
        if (block.rank > 0) { //NB: avoid multiple reduction declarations
            for (const InstrPtr instr: block._sweeps) {
                openmp_reductions.push_back(instr);
            }
        }
    }

    //Let's write the OpenMP reductions
    for (const InstrPtr instr: openmp_reductions) {
        assert(bh_noperands(instr->opcode) == 3);
        bh_base *base = instr->operand[0].base;
        ss << " reduction(" << openmp_reduce_symbol(instr->opcode) << ":";
        ss << (base_ids.isScalarReplaced(base)?"s":"t");
        ss << base_ids[base] << ")";
    }
    const string ss_str = ss.str();
    if(not ss_str.empty()) {
        out << "#pragma omp" << ss_str << "\n";
        spaces(out, 4 + block.rank*4);
    }
}

// Writes the OpenMP specific for-loop header
void loop_head_writer(BaseDB &base_ids, const LoopB &block, const ConfigParser &config, bool loop_is_peeled,
                      const vector<const LoopB *> &threaded_blocks, stringstream &out) {

    // Let's write the OpenMP loop header
    {
        int64_t for_loop_size = block.size;
        if (block._sweeps.size() > 0 and loop_is_peeled) // If the for-loop has been peeled, its size is one less
            --for_loop_size;
        // No need to parallel one-sized loops
        if (for_loop_size > 1) {
            write_openmp_header(block, base_ids, config, out);
        }
    }

    // Write the for-loop header
    string itername;
    {stringstream t; t << "i" << block.rank; itername = t.str();}
    out << "for(uint64_t " << itername;
    if (block._sweeps.size() > 0 and loop_is_peeled) // If the for-loop has been peeled, we should start at 1
        out << "=1; ";
    else
        out << "=0; ";
    out << itername << " < " << block.size << "; ++" << itername << ") {\n";
}

// Remove empty blocks inplace
void remove_empty_blocks(vector<Block> &block_list) {
    for (size_t i=0; i < block_list.size(); ) {
        Block &b = block_list[i];
        if (b.isInstr()) {
            ++i;
        } else if (b.isSystemOnly()) {
            block_list.erase(block_list.begin()+i);
        } else {
            remove_empty_blocks(b.getLoop()._block_list);
            ++i;
        }
    }
}

void write_kernel(Kernel &kernel, BaseDB &base_ids, const ConfigParser &config, stringstream &ss) {

    // Make sure all arrays are allocated
    for (bh_base *base: kernel.getNonTemps()) {
        bh_data_malloc(base);
    }

    // Write the need includes
    ss << "#include <stdint.h>\n";
    ss << "#include <stdlib.h>\n";
    ss << "#include <stdbool.h>\n";
    ss << "#include <complex.h>\n";
    ss << "#include <tgmath.h>\n";
    ss << "#include <math.h>\n";
    if (kernel.useRandom()) { // Write the random function
        ss << "#include <kernel_dependencies/random123_openmp.h>\n";
    }
    ss << "\n";

    // Write the header of the execute function
    ss << "void execute(";
    for(size_t i=0; i < kernel.getNonTemps().size(); ++i) {
        bh_base *b = kernel.getNonTemps()[i];
        ss << write_c99_type(b->type) << " a" << base_ids[b] << "[static " << b->nelem << "]";
        if (i+1 < kernel.getNonTemps().size()) {
            ss << ", ";
        }
    }
    ss << ") {\n";

    // Write the block that makes up the body of 'execute()'
    write_loop_block(base_ids, kernel.block, config, {}, false, write_c99_type, loop_head_writer, ss);

    ss << "}\n\n";

    // Write the launcher function, which will convert the data_list of void pointers
    // to typed arrays and call the execute function
    {
        ss << "void launcher(void* data_list[]) {\n";
        for(size_t i=0; i < kernel.getNonTemps().size(); ++i) {
            spaces(ss, 4);
            bh_base *b = kernel.getNonTemps()[i];
            ss << write_c99_type(b->type) << " *a" << base_ids[b];
            ss << " = data_list[" << i << "];\n";
        }
        spaces(ss, 4);
        ss << "execute(";
        for(size_t i=0; i < kernel.getNonTemps().size(); ++i) {
            bh_base *b = kernel.getNonTemps()[i];
            ss << "a" << base_ids[b];
            if (i+1 < kernel.getNonTemps().size()) {
                ss << ", ";
            }
        }
        ss << ");\n";
        ss << "}\n";
    }
}

// Sets the constructor flag of each instruction in 'instr_list'
void set_constructor_flag(vector<bh_instruction> &instr_list) {
    set<bh_base*> initiated; // Arrays initiated in 'instr_list'
    for(bh_instruction &instr: instr_list) {
        instr.constructor = false;
        int nop = bh_noperands(instr.opcode);
        for (bh_intp o = 0; o < nop; ++o) {
            const bh_view &v = instr.operand[o];
            if (not bh_is_constant(&v)) {
                assert(v.base != NULL);
                if (v.base->data == NULL and not util::exist(initiated, v.base)) {
                    if (o == 0) { // It is only the output that is initiated
                        initiated.insert(v.base);
                        instr.constructor = true;
                    }
                }
            }
        }
    }
}

void Impl::execute(bh_ir *bhir) {
    auto texecution = chrono::steady_clock::now();

    // Let's start by extracting a clean list of instructions from the 'bhir'
    vector<bh_instruction*> instr_list = remove_non_computed_system_instr(bhir->instr_list);

    // Set the constructor flag
    if (config.defaultGet<bool>("array_contraction", true)) {
        set_constructor_flag(bhir->instr_list);
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

        } else {
            // Let's fuse the 'instr_list' into blocks
            block_list = fuser_singleton(instr_list);
            if (config.defaultGet<bool>("serial_fusion", false)) {
                fuser_serial(block_list, 1);
            } else {
                fuser_greedy(block_list);
                block_list = collapse_redundant_axes(block_list);
            }
            remove_empty_blocks(block_list);
            fcache.insert(instr_list, block_list);
        }
    }

    // Pretty printing the block
    if (config.defaultGet<bool>("graph", false)) {
        graph::DAG dag = graph::from_block_list(block_list);
        graph::pprint(dag, "dag");
    }

    // Some statistics
    time_fusion += chrono::steady_clock::now() - texecution;
    if (config.defaultGet<bool>("prof", false)) {
        for (const bh_instruction *instr: instr_list) {
            if (not bh_opcode_is_system(instr->opcode)) {
                totalwork += bh_nelements(instr->operand[0]);
            }
        }
    }

    for(const Block &block: block_list) {
        assert(not block.isInstr());

        //Let's create a kernel
        Kernel kernel(block.getLoop());

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
        if (config.defaultGet<bool>("verbose", false))
            cout << kernel.block;

        // Code generation
        stringstream ss;
        write_kernel(kernel, base_ids, config, ss);

        // Compile the kernel
        auto tbuild = chrono::steady_clock::now();
        KernelFunction func = _store.getFunction(ss.str());
        assert(func != NULL);
        time_build += chrono::steady_clock::now() - tbuild;

        // Create a 'data_list' of data pointers
        vector<void*> data_list;
        data_list.reserve(kernel.getNonTemps().size());
        for(bh_base *base: kernel.getNonTemps()) {
            assert(base->data != NULL);
            data_list.push_back(base->data);
        }

        auto texec = chrono::steady_clock::now();
        // Call the launcher function with the 'data_list', which will execute the kernel
        func(&data_list[0]);
        time_exec += chrono::steady_clock::now() - texec;

        // Finally, let's cleanup
        for(bh_base *base: kernel.getFrees()) {
            bh_data_free(base);
        }
    }
    time_total_execution += chrono::steady_clock::now() - texecution;
}

