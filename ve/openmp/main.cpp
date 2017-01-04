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
#include <jitk/type.hpp>
#include <jitk/graph.hpp>
#include <jitk/transformer.hpp>
#include <jitk/fuser_cache.hpp>

#include "store.hpp"

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

namespace {
void spaces(stringstream &out, int num) {
    for (int i = 0; i < num; ++i) {
        out << " ";
    }
}
}

Impl::~Impl() {
    if (config.defaultGet<bool>("prof", false)) {
        const int64_t store_hits = _store.num_lookups - _store.num_lookup_misses;
        const uint64_t fcache_hits = fcache.num_lookups - fcache.num_lookup_misses;
        cout << "[VE-OPENMP] Profiling: " << endl;
        cout << "\tKernel Store Hits:   " << store_hits << "/" << _store.num_lookups \
                                          << " (" << 100.0*store_hits/_store.num_lookups << "%)" << endl;
        cout << "\tFuse Cache hits:     " << fcache_hits << "/" << fcache.num_lookups \
                                          << " (" << 100.0*fcache_hits/fcache.num_lookups << "%)" << endl;
        cout << "\tArray contractions:  " << num_temp_arrays << "/" << num_base_arrays \
                                          << " (" << 100.0*num_temp_arrays/num_base_arrays << "%)" << endl;
        cout << "\tTotal Work: " << (double) totalwork << " operations" << endl;
        cout << "\tTotal Execution:  " << time_total_execution.count() << "s" << endl;
        cout << "\t  Fusion: " << time_fusion.count() << "s" << endl;
        cout << "\t  Build:  " << time_build.count() << "s" << endl;
        cout << "\t  Exec:   " << time_exec.count() << "s" << endl;
    }
}

// Return the OpenMP reduction symbol
const char* openmp_reduce_symbol(bh_opcode opcode) {
    switch (opcode) {
        case BH_ADD_REDUCE:
            return "+";
        case BH_MULTIPLY_REDUCE:
            return "*";
        case BH_BITWISE_AND_REDUCE:
            return "&";
        case BH_BITWISE_OR_REDUCE:
            return "|";
        case BH_BITWISE_XOR_REDUCE:
            return "^";
        case BH_MAXIMUM_REDUCE:
            return "max";
        case BH_MINIMUM_REDUCE:
            return "min";
        default:
            return NULL;
    }
}

// Is 'opcode' compatible with OpenMP reductions such as reduction(+:var)
bool openmp_reduce_compatible(bh_opcode opcode) {
    return openmp_reduce_symbol(opcode) != NULL;
}

// Is the 'block' compatible with OpenMP
bool openmp_compatible(const LoopB &block) {
    // For now, all sweeps must be reductions
    for (const InstrPtr instr: block._sweeps) {
        if (not bh_opcode_is_reduction(instr->opcode)) {
            return false;
        }
    }
    return true;
}

// Is the 'block' compatible with OpenMP SIMD
bool simd_compatible(const LoopB &block, const BaseDB &base_ids) {

    // Check for non-compatible reductions
    for (const InstrPtr instr: block._sweeps) {
        if (not openmp_reduce_compatible(instr->opcode))
            return false;
    }

    // An OpenMP SIMD loop does not support ANY OpenMP pragmas
    for (const bh_base* b: block.getAllBases()) {
        if (base_ids.isOpenmpAtomic(b) or base_ids.isOpenmpCritical(b))
            return false;
    }
    return true;
}

// Does 'opcode' support the OpenMP Atomic guard?
bool openmp_atomic_compatible(bh_opcode opcode) {
    switch (opcode) {
        case BH_ADD_REDUCE:
        case BH_MULTIPLY_REDUCE:
        case BH_BITWISE_AND_REDUCE:
        case BH_BITWISE_OR_REDUCE:
        case BH_BITWISE_XOR_REDUCE:
            return true;
        default:
            return false;
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

// Does 'instr' reduce over the innermost axis?
// Notice, that such a reduction computes each output element completely before moving
// to the next element.
bool sweeping_innermost_axis(InstrPtr instr) {
    if (not bh_opcode_is_sweep(instr->opcode))
        return false;
    assert(bh_noperands(instr->opcode) == 3);
    return instr->sweep_axis() == instr->operand[1].ndim-1;
}

void write_loop_block(BaseDB &base_ids, const LoopB &block, const ConfigParser &config, stringstream &out) {
    spaces(out, 4 + block.rank*4);

    // Let's scalar replace reduction outputs that reduces over the innermost axis
    vector<bh_view> scalar_replacements;
    for (const InstrPtr instr: block._sweeps) {
        if (bh_opcode_is_reduction(instr->opcode) and sweeping_innermost_axis(instr)) {
            bh_base *base = instr->operand[0].base;
            if (base_ids.isTmp(base) or base_ids.isLocallyDeclared(base))
                continue;
            scalar_replacements.push_back(instr->operand[0]);
            base_ids.insertScalarReplacement(base);
            // Let's write the declaration of the scalar variable
            base_ids.writeDeclaration(base, write_type(base->type), out);
            out << "\n";
            spaces(out, 4 + block.rank * 4);
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
            if (not base_ids.isArray(base) and not base_ids.isLocallyDeclared(base)) {
                base_ids.writeDeclaration(base, write_type(base->type), out);
                out << "\n";
                spaces(out, 4 + block.rank * 4);
            }
            base_ids.getName(base, out);
            out << " = ";
            write_reduce_identity(instr->opcode, base->type, out);
            out << ";\n";
            spaces(out, 4 + block.rank * 4);
        }
    }

    // All local temporary arrays needs an variable declaration
    const set<bh_base*> local_tmps = block.getLocalTemps();

    // If this block is sweeped, we will "peel" the for-loop such that the
    // sweep instruction is replaced with BH_IDENTITY in the first iteration
    if (block._sweeps.size() > 0 and need_to_peel) {
        BaseDB base_ids_tmp(base_ids);
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
        out << "{ // Peeled loop, 1. sweep iteration\n";
        spaces(out, 8 + block.rank*4);
        out << "uint64_t " << itername << " = 0;\n";
        // Write temporary array declarations
        for (bh_base* base: local_tmps) {
            assert(base_ids_tmp.isTmp(base));
            if (not base_ids_tmp.isLocallyDeclared(base)) {
                spaces(out, 8 + block.rank * 4);
                base_ids_tmp.writeDeclaration(base, write_type(base->type), out);
                out << "\n";
            }
        }
        out << "\n";
        for (const Block &b: peeled_block._block_list) {
            if (b.isInstr()) {
                if (b.getInstr() != NULL) {
                    spaces(out, 4 + b.rank()*4);
                    write_instr(base_ids_tmp, *b.getInstr(), out, false);
                }
            } else {
                write_loop_block(base_ids_tmp, b.getLoop(), config, out);
            }
        }
        spaces(out, 4 + block.rank*4);
        out << "}" << endl;
        spaces(out, 4 + block.rank*4);
    }

    // Let's write the OpenMP loop header
    {
        int64_t for_loop_size = block.size;
        if (block._sweeps.size() > 0 and need_to_peel) // If the for-loop has been peeled, its size is one less
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
    if (block._sweeps.size() > 0 and need_to_peel) // If the for-loop has been peeled, we should start at 1
        out << "=1; ";
    else
        out << "=0; ";
    out << itername << " < " << block.size << "; ++" << itername << ") {\n";

    // Write temporary array declarations
    for (bh_base* base: local_tmps) {
        assert(base_ids.isTmp(base));
        if (not base_ids.isLocallyDeclared(base)) {
            spaces(out, 8 + block.rank * 4);
            base_ids.writeDeclaration(base, write_type(base->type), out);
            out << "\n";
        }
    }

    // Write the for-loop body
    for (const Block &b: block._block_list) {
        if (b.isInstr()) { // Finally, let's write the instruction
            const InstrPtr instr = b.getInstr();
            if (bh_noperands(instr->opcode) > 0 and not bh_opcode_is_system(instr->opcode)) {
                if (base_ids.isOpenmpAtomic(instr->operand[0].base)) {
                    spaces(out, 4 + b.rank()*4);
                    out << "#pragma omp atomic\n";
                } else if (base_ids.isOpenmpCritical(instr->operand[0].base)) {
                    spaces(out, 4 + b.rank()*4);
                    out << "#pragma omp critical\n";
                }
            }
            spaces(out, 4 + b.rank()*4);
            write_instr(base_ids, *instr, out);
        } else {
            write_loop_block(base_ids, b.getLoop(), config, out);
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
        ss << write_type(b->type) << " a" << base_ids[b] << "[static " << b->nelem << "]";
        if (i+1 < kernel.getNonTemps().size()) {
            ss << ", ";
        }
    }
    ss << ") {\n";

    // Write the block that makes up the body of 'execute()'
    write_loop_block(base_ids, kernel.block, config, ss);
    
    ss << "}\n\n";

    // Write the launcher function, which will convert the data_list of void pointers
    // to typed arrays and call the execute function
    {
        ss << "void launcher(void* data_list[]) {" << endl;
        for(size_t i=0; i < kernel.getNonTemps().size(); ++i) {
            spaces(ss, 4);
            bh_base *b = kernel.getNonTemps()[i];
            ss << write_type(b->type) << " *a" << base_ids[b];
            ss << " = data_list[" << i << "];" << endl;
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
        ss << ");" << endl;
        ss << "}" << endl;
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

