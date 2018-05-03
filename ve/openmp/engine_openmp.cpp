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

#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <iomanip>
#include <dlfcn.h>
#include <jitk/codegen_util.hpp>
#include <jitk/compiler.hpp>
#include <jitk/fuser_cache.hpp>
#include <jitk/codegen_cache.hpp>
#include <jitk/block.hpp>
#include <thread>

#include <bh_util.hpp>
#include "engine_openmp.hpp"
#include "openmp_util.hpp"

using namespace std;
using namespace bohrium::jitk;
namespace fs = boost::filesystem;

namespace bohrium {

EngineOpenMP::EngineOpenMP(component::ComponentVE &comp, jitk::Statistics &stat) :
        EngineCPU(comp, stat),
        compiler(comp.config.get<string>("compiler_cmd"), verbose, comp.config.file_dir.string()) {

    compilation_hash = util::hash(compiler.cmd_template);

    // Initiate cache limits
    const uint64_t sys_mem = bh_main_memory_total();
    malloc_cache_limit_in_percent = comp.config.defaultGet<int64_t>("malloc_cache_limit", 80);
    if (malloc_cache_limit_in_percent < 0 or malloc_cache_limit_in_percent > 100) {
        throw std::runtime_error("config: `malloc_cache_limit` must be between 0 and 100");
    }
    malloc_cache_limit_in_bytes = static_cast<int64_t>(std::floor(sys_mem * (malloc_cache_limit_in_percent / 100.0)));
    bh_set_malloc_cache_limit(static_cast<uint64_t>(malloc_cache_limit_in_bytes));
}

EngineOpenMP::~EngineOpenMP() {
    // Move JIT kernels to the cache dir
    if (not cache_bin_dir.empty()) {
        try {
            for (const auto &kernel: _functions) {
                const fs::path src = tmp_bin_dir / jitk::hash_filename(compilation_hash, kernel.first, ".so");
                if (fs::exists(src)) {
                    const fs::path dst = cache_bin_dir / jitk::hash_filename(compilation_hash, kernel.first, ".so");
                    if (not fs::exists(dst)) {
                        fs::copy_file(src, dst);
                    }
                }
            }
        } catch (const boost::filesystem::filesystem_error &e) {
            cout << "Warning: couldn't write JIT kernels to disk to " << cache_bin_dir
                 << ". " << e.what() << endl;
        }
    }

    // File clean up
    if (not verbose) {
        fs::remove_all(tmp_src_dir);
    }

    if (cache_file_max != -1 and not cache_bin_dir.empty()) {
        util::remove_old_files(cache_bin_dir, cache_file_max);
    }

    // If this cleanup is enabled, the application segfaults
    // on destruction of the EngineOpenMP class.
    //
    // The reason for that is that OpenMP currently has no
    // means in the definition of the standard to do a proper
    // cleanup and hence internally does something funny.
    // See https://stackoverflow.com/questions/5200418/destroying-threads-in-openmp-c
    // for details.
    //
    // TODO A more sensible way to get around this issue
    //      would be to also dlopen the openmp library
    //      itself and therefore increase our reference count
    //      to it. This enables the kernel so files to be unlinked
    //      and cleaned up, but prevents the problematic code on
    //      openmp cleanup to be triggered at bohrium runtime.

    // for(void *handle: _lib_handles) {
    //     dlerror(); // Reset errors
    //     if (dlclose(handle)) {
    //         cerr << dlerror() << endl;
    //     }
    // }
}

KernelFunction EngineOpenMP::getFunction(const string &source, const std::string &func_name) {
    uint64_t hash = util::hash(source);
    ++stat.kernel_cache_lookups;

    // Do we have the function compiled and ready already?
    if (_functions.find(hash) != _functions.end()) {
        return _functions.at(hash);
    }

    fs::path binfile = cache_bin_dir / jitk::hash_filename(compilation_hash, hash, ".so");

    // If the binary file of the kernel doesn't exist we create it
    if (verbose or cache_bin_dir.empty() or not fs::exists(binfile)) {
        ++stat.kernel_cache_misses;

        // We create the binary file in the tmp dir
        binfile = tmp_bin_dir / jitk::hash_filename(compilation_hash, hash, ".so");

        // Write the source file and compile it (reading from disk)
        // NB: this is a nice debug option, but will hurt performance
        if (verbose) {
            std::string source_filename = jitk::hash_filename(compilation_hash, hash, ".c");
            fs::path srcfile = jitk::write_source2file(source, tmp_src_dir, source_filename, true);
            compiler.compile(binfile.string(), srcfile.string());
        } else {
            // Pipe the source directly into the compiler thus no source file is written
            compiler.compile(binfile.string(), source.c_str(), source.size());
        }
    }

    // Load the shared library
    void *lib_handle = dlopen(binfile.string().c_str(), RTLD_NOW);
    if (lib_handle == nullptr) {
        cerr << "Cannot load library: " << dlerror() << endl;
        throw runtime_error("VE-OPENMP: Cannot load library");
    }
    _lib_handles.push_back(lib_handle);

    // Load the launcher function
    // The (clumsy) cast conforms with the ISO C standard and will
    // avoid any compiler warnings.
    dlerror(); // Reset errors
    *(void **) (&_functions[hash]) = dlsym(lib_handle, func_name.c_str());
    const char *dlsym_error = dlerror();
    if (dlsym_error != nullptr) {
        cerr << "Cannot load function launcher(): " << dlsym_error << endl;
        throw runtime_error("VE-OPENMP: Cannot load function launcher()");
    }
    return _functions.at(hash);
}


void EngineOpenMP::execute(const jitk::SymbolTable &symbols,
                           const std::string &source,
                           uint64_t codegen_hash,
                           const std::vector<const bh_instruction *> &constants) {
    // Notice, we use a "pure" hash of `source` to make sure that the `source_filename` always
    // corresponds to `source` even if `codegen_hash` is buggy.
    uint64_t hash = util::hash(source);
    std::string source_filename = jitk::hash_filename(compilation_hash, hash, ".c");

    // Make sure all arrays are allocated
    for (bh_base *base: symbols.getParams()) {
        bh_data_malloc(base);
    }

    // Compile the kernel
    auto tbuild = chrono::steady_clock::now();
    string func_name;
    {
        stringstream t;
        t << "launcher_" << codegen_hash;
        func_name = t.str();
    }
    KernelFunction func = getFunction(source, func_name);
    assert(func != nullptr);
    stat.time_compile += chrono::steady_clock::now() - tbuild;

    // Create a 'data_list' of data pointers
    vector<void *> data_list;
    data_list.reserve(symbols.getParams().size());
    for (bh_base *base: symbols.getParams()) {
        assert(base->data != nullptr);
        data_list.push_back(base->data);
    }

    // And the offset-and-strides
    vector<uint64_t> offset_and_strides;
    offset_and_strides.reserve(symbols.offsetStrideViews().size());
    for (const bh_view *view: symbols.offsetStrideViews()) {
        const uint64_t t = (uint64_t) view->start;
        offset_and_strides.push_back(t);
        for (int i = 0; i < view->ndim; ++i) {
            const uint64_t s = (uint64_t) view->stride[i];
            offset_and_strides.push_back(s);
        }
    }

    // And the constants
    vector<bh_constant_value> constant_arg;
    constant_arg.reserve(constants.size());
    for (const bh_instruction *instr: constants) {
        constant_arg.push_back(instr->constant.value);
    }

    auto start_exec = chrono::steady_clock::now();
    // Call the launcher function, which will execute the kernel
    func(&data_list[0], &offset_and_strides[0], &constant_arg[0]);
    auto texec = chrono::steady_clock::now() - start_exec;
    stat.time_exec += texec;
    stat.time_per_kernel[source_filename].register_exec_time(texec);

}

// Writes the OpenMP specific for-loop header
void EngineOpenMP::loopHeadWriter(const jitk::SymbolTable &symbols,
                                  jitk::Scope &scope,
                                  const jitk::LoopB &block,
                                  bool loop_is_peeled,
                                  const vector<uint64_t> &thread_stack,
                                  stringstream &out) {
    // Let's write the OpenMP loop header
    int64_t for_loop_size = block.size;
    // If the for-loop has been peeled, its size is one less
    if (block._sweeps.size() > 0 and loop_is_peeled) {
        --for_loop_size;
    }
    // No need to parallel one-sized loops
    if (for_loop_size > 1) {
        writeHeader(symbols, scope, block, out);
    }
    // Write the for-loop header
    string itername;
    {
        stringstream t;
        t << "i" << block.rank;
        itername = t.str();
    }
    out << "for(uint64_t " << itername;
    if (block._sweeps.size() > 0 and loop_is_peeled) {
        // If the for-loop has been peeled, we should start at 1
        out << " = 1; ";
    } else {
        out << " = 0; ";
    }
    out << itername << " < " << block.size << "; ++" << itername << ") {\n";
}

// Writing the OpenMP header, which include "parallel for" and "simd"
void EngineOpenMP::writeHeader(const jitk::SymbolTable &symbols,
                               jitk::Scope &scope,
                               const jitk::LoopB &block,
                               std::stringstream &out) {
    if (not comp.config.defaultGet<bool>("compiler_openmp", false)) {
        return;
    }
    const bool enable_simd = comp.config.defaultGet<bool>("compiler_openmp_simd", false);

    // All reductions that can be handle directly be the OpenMP header e.g. reduction(+:var)
    std::vector<jitk::InstrPtr> openmp_reductions;

    // Order all sweep instructions by the viewID of their first operand.
    // This makes the source of the kernels more identical, which improve the code and compile caches.
    const std::vector<jitk::InstrPtr> ordered_block_sweeps = order_sweep_set(block._sweeps, symbols);

    stringstream ss;
    // "OpenMP for" goes to the outermost loop
    if (block.rank == 0 and openmp_compatible(block)) {
        ss << " parallel for";
        // Since we are doing parallel for, we should either do OpenMP reductions or protect the sweep instructions
        for (const jitk::InstrPtr &instr: ordered_block_sweeps) {
            assert(instr->operand.size() == 3);
            const bh_view &view = instr->operand[0];
            if (openmp_reduce_compatible(instr->opcode) and (scope.isScalarReplaced(view) or scope.isTmp(view.base))) {
                openmp_reductions.push_back(instr);
            } else if (openmp_atomic_compatible(instr->opcode)) {
                scope.insertOpenmpAtomic(instr);
            } else {
                scope.insertOpenmpCritical(instr);
            }
        }
    }

    // "OpenMP SIMD" goes to the innermost loop (which might also be the outermost loop)
    if (enable_simd and block.isInnermost() and simd_compatible(block, scope)) {
        ss << " simd";
        if (block.rank > 0) { // NB: avoid multiple reduction declarations
            for (const jitk::InstrPtr &instr: ordered_block_sweeps) {
                openmp_reductions.push_back(instr);
            }
        }
    }

    //Let's write the OpenMP reductions
    for (const jitk::InstrPtr &instr: openmp_reductions) {
        assert(instr->operand.size() == 3);
        ss << " reduction(" << openmp_reduce_symbol(instr->opcode) << ":";
        scope.getName(instr->operand[0], ss);
        ss << ")";
    }
    const string ss_str = ss.str();
    if (not ss_str.empty()) {
        out << "#pragma omp" << ss_str << "\n";
        util::spaces(out, 4 + block.rank * 4);
    }
}


void EngineOpenMP::writeBlock(const jitk::SymbolTable &symbols,
                              const jitk::Scope *parent_scope,
                              const jitk::LoopB &kernel,
                              const std::vector<uint64_t> &thread_stack,
                              bool opencl,
                              std::stringstream &out) {

    if (kernel.isSystemOnly()) {
        out << "// Removed loop with only system instructions\n";
        return;
    }

    std::set<jitk::InstrPtr> sweeps_in_child;
    for (const jitk::Block &sub_block: kernel._block_list) {
        if (not sub_block.isInstr()) {
            sweeps_in_child.insert(sub_block.getLoop()._sweeps.begin(), sub_block.getLoop()._sweeps.end());
        }
    }
    // Order all sweep instructions by the viewID of their first operand.
    // This makes the source of the kernels more identical, which improve the code and compile caches.
    const vector<jitk::InstrPtr> ordered_block_sweeps = order_sweep_set(sweeps_in_child, symbols);

    // Let's find the local temporary arrays and the arrays to scalar replace
    const set<bh_base *> &local_tmps = kernel.getLocalTemps();

    // We always scalar replace reduction outputs that reduces over the innermost axis
    vector<const bh_view *> scalar_replaced_reduction_outputs;
    for (const jitk::InstrPtr &instr: ordered_block_sweeps) {
        if (bh_opcode_is_reduction(instr->opcode) and jitk::sweeping_innermost_axis(instr)) {
            if (local_tmps.find(instr->operand[0].base) == local_tmps.end()) {
                scalar_replaced_reduction_outputs.push_back(&instr->operand[0]);
            }
        }
    }

    // Let's scalar replace input-only arrays that are used multiple times
    vector<const bh_view *> srio = jitk::scalar_replaced_input_only(kernel, parent_scope, local_tmps);
    jitk::Scope scope(symbols, parent_scope, local_tmps, scalar_replaced_reduction_outputs, srio);

    // Write temporary and scalar replaced array declarations
    vector<const bh_view*> scalar_replaced_to_write_back;
    for (const jitk::Block &block: kernel._block_list) {
        if (block.isInstr()) {
            const jitk::InstrPtr instr = block.getInstr();
            for (const bh_view *view: instr->get_views()) {
                if (not scope.isDeclared(*view)) {
                    if (scope.isTmp(view->base)) {
                        util::spaces(out, 8 + kernel.rank * 4);
                        scope.writeDeclaration(*view, writeType(view->base->type), out);
                        out << "\n";
                    } else if (scope.isScalarReplaced(*view)) {
                        util::spaces(out, 8 + kernel.rank * 4);
                        scope.writeDeclaration(*view, writeType(view->base->type), out);
                        out << " " << scope.getName(*view) << " = a" << symbols.baseID(view->base);
                        write_array_subscription(scope, *view, out);
                        out << ";";
                        out << "\n";
                        if (scope.isScalarReplaced_RW(view->base)) {
                            scalar_replaced_to_write_back.push_back(view);
                        }
                    }
                }
            }
        }
    }

    //Let's declare indexes if we are not at the kernel level (rank == -1)
    if (kernel.rank >= 0) {
        for (const jitk::Block &block: kernel._block_list) {
            if (block.isInstr()) {
                const jitk::InstrPtr instr = block.getInstr();
                for (const bh_view *view: instr->get_views()) {
                    if (symbols.existIdxID(*view) and scope.isArray(*view)) {
                        if (not scope.isIdxDeclared(*view)) {
                            util::spaces(out, 8 + kernel.rank * 4);
                            scope.writeIdxDeclaration(*view, writeType(bh_type::UINT64), out);
                            out << "\n";
                        }
                    }
                }
            }
        }
    }

    for (const Block &b: kernel._block_list) {
        if (b.isInstr()) { // Finally, let's write the instruction
            const InstrPtr instr = b.getInstr();
            if (not bh_opcode_is_system(instr->opcode)) {
                if (instr->operand.size() > 0) {
                    if (scope.isOpenmpAtomic(instr)) {
                        util::spaces(out, 4 + b.rank() * 4);
                        out << "#pragma omp atomic\n";
                    } else if (scope.isOpenmpCritical(instr)) {
                        util::spaces(out, 4 + b.rank() * 4);
                        out << "#pragma omp critical\n";
                    }
                }
                util::spaces(out, 4 + b.rank() * 4);
                write_instr(scope, *instr, out);
            }
        } else {
            util::spaces(out, 4 + b.rank() * 4);
            loopHeadWriter(symbols, scope, b.getLoop(), false, thread_stack, out);
            writeBlock(symbols, &scope, b.getLoop(), thread_stack, opencl, out);
            util::spaces(out, 4 + b.rank() * 4);
            out << "}\n";
        }
    }

    // Let's copy the scalar replaced reduction outputs back to the original array
    for (const bh_view *view: scalar_replaced_to_write_back) {
        util::spaces(out, 8 + kernel.rank * 4);
        out << "a" << symbols.baseID(view->base);
        write_array_subscription(scope, *view, out, true);
        out << " = ";
        scope.getName(*view, out);
        out << ";\n";
    }
}

void EngineOpenMP::writeKernel(const jitk::Block &block,
                               const jitk::SymbolTable &symbols,
                               const std::vector<bh_base *> &kernel_temps,
                               uint64_t codegen_hash,
                               std::stringstream &ss) {

    assert(block.rank() == -1);
    assert(not block.isInstr());

    // Write the need includes
    ss << "#include <stdint.h>\n";
    ss << "#include <stdlib.h>\n";
    ss << "#include <stdbool.h>\n";
    ss << "#include <complex.h>\n";
    ss << "#include <tgmath.h>\n";
    ss << "#include <math.h>\n";
    if (symbols.useRandom()) { // Write the random function
        ss << "#include <kernel_dependencies/random123_openmp.h>\n";
    }
    writeUnionType(ss); // We always need to declare the union of all constant data types
    ss << "\n";

    // Write the header of the execute function
    ss << "void execute_" << codegen_hash;
    writeKernelFunctionArguments(symbols, ss, nullptr);

    // Write the block that makes up the body of 'execute()'
    ss << "{\n";
    // Write allocations of the kernel temporaries
    for (const bh_base *b: kernel_temps) {
        util::spaces(ss, 4);
        ss << writeType(b->type) << " * __restrict__ a" << symbols.baseID(b) << " = malloc(" << b->nbytes() << ");\n";
    }
    ss << "\n";

    writeBlock(symbols, nullptr, block.getLoop(), {}, false, ss);

    // Write frees of the kernel temporaries
    ss << "\n";
    for (const bh_base *b: kernel_temps) {
        util::spaces(ss, 4);
        ss << "free(" << "a" << symbols.baseID(b) << ");\n";
    }
    ss << "}\n\n";

    // Write the launcher function, which will convert the data_list of void pointers
    // to typed arrays and call the execute function
    {
        ss << "void launcher_" << codegen_hash
           << "(void* data_list[], uint64_t offset_strides[], union dtype constants[]) {\n";
        for (size_t i = 0; i < symbols.getParams().size(); ++i) {
            util::spaces(ss, 4);
            bh_base *b = symbols.getParams()[i];
            ss << writeType(b->type) << " *a" << symbols.baseID(b);
            ss << " = data_list[" << i << "];\n";
        }

        util::spaces(ss, 4);
        ss << "execute_" << codegen_hash << "(";

        // We create the comma separated list of args and saves it in `stmp`
        stringstream stmp;
        for (size_t i = 0; i < symbols.getParams().size(); ++i) {
            bh_base *b = symbols.getParams()[i];
            stmp << "a" << symbols.baseID(b) << ", ";
        }

        uint64_t count = 0;
        for (const bh_view *view: symbols.offsetStrideViews()) {
            stmp << "offset_strides[" << count++ << "], ";
            for (int i = 0; i < view->ndim; ++i) {
                stmp << "offset_strides[" << count++ << "], ";
            }
        }

        if (not symbols.constIDs().empty()) {
            uint64_t i = 0;
            for (auto it = symbols.constIDs().begin(); it != symbols.constIDs().end(); ++it) {
                const jitk::InstrPtr &instr = *it;
                stmp << "constants[" << i++ << "]." << bh_type_text(instr->constant.type) << ", ";
            }
        }

        // And then we write `stmp` into `ss` excluding the last comma
        const string strtmp = stmp.str();
        if (not strtmp.empty()) {
            ss << strtmp.substr(0, strtmp.size() - 2);
        }
        ss << ");\n";
        ss << "}\n";
    }
}

std::string EngineOpenMP::info() const {
    stringstream ss;
    ss << std::boolalpha; // Printing true/false instead of 1/0
    ss << "----" << "\n";
    ss << "OpenMP:" << "\n";
    ss << "  Main memory: " << bh_main_memory_total() / 1024 / 1024 << " MB\n";
    ss << "  Hardware threads: " << std::thread::hardware_concurrency() << "\n";
    ss << "  Malloc cache limit: " << malloc_cache_limit_in_bytes / 1024 / 1024
       << " MB (" << malloc_cache_limit_in_percent << "%)\n";
    ss << "  Cache dir: " << comp.config.defaultGet<string>("cache_dir", "") << "\n";
    ss << "  Temp dir: " << jitk::get_tmp_path(comp.config) << "\n";

    ss << "  Codegen flags:\n";
    ss << "    OpenMP: " << comp.config.defaultGet<bool>("compiler_openmp", false) << "\n";
    ss << "    OpenMP+SIMD: " << comp.config.defaultGet<bool>("compiler_openmp_simd", false) << "\n";
    ss << "    Index-as-var: " << comp.config.defaultGet<bool>("index_as_var", true) << "\n";
    ss << "    Strides-as-var: " << comp.config.defaultGet<bool>("strides_as_var", true) << "\n";
    ss << "    Const-as-var: " << comp.config.defaultGet<bool>("const_as_var", true) << "\n";

    ss << "  JIT Command: \"" << compiler.cmd_template << "\"\n";
    return ss.str();
}

// Return C99 types, which are used inside the C99 kernels
const std::string EngineOpenMP::writeType(bh_type dtype) {
    switch (dtype) {
        case bh_type::BOOL:
            return "bool";
        case bh_type::INT8:
            return "int8_t";
        case bh_type::INT16:
            return "int16_t";
        case bh_type::INT32:
            return "int32_t";
        case bh_type::INT64:
            return "int64_t";
        case bh_type::UINT8:
            return "uint8_t";
        case bh_type::UINT16:
            return "uint16_t";
        case bh_type::UINT32:
            return "uint32_t";
        case bh_type::UINT64:
            return "uint64_t";
        case bh_type::FLOAT32:
            return "float";
        case bh_type::FLOAT64:
            return "double";
        case bh_type::COMPLEX64:
            return "float complex";
        case bh_type::COMPLEX128:
            return "double complex";
        case bh_type::R123:
            return "r123_t"; // Defined by `write_c99_dtype_union()`
        default:
            std::cerr << "Unknown C99 type: " << bh_type_text(dtype) << std::endl;
            throw std::runtime_error("Unknown C99 type");
    }
}

} // bohrium
