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
#include <mpi.h>

#include <bh_util.hpp>
#include "engine_pgas.hpp"

using namespace std;
using namespace bohrium::jitk;
namespace fs = boost::filesystem;

namespace bohrium {

EnginePGAS::EnginePGAS(component::ComponentVE &comp, jitk::Statistics &stat) :
        EngineCPU(comp, stat),
        compiler(comp.config.get<string>("compiler_cmd"), comp.config.file_dir.string(), verbose) {

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

EnginePGAS::~EnginePGAS() {
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
    // on destruction of the EnginePGAS class.
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

KernelFunction EnginePGAS::getFunction(const string &source, const std::string &func_name) {
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


void EnginePGAS::execute(const jitk::SymbolTable &symbols,
                         const std::string &source,
                         uint64_t codegen_hash,
                         const std::vector<const bh_instruction *> &constants) {
    // Notice, we use a "pure" hash of `source` to make sure that the `source_filename` always
    // corresponds to `source` even if `codegen_hash` is buggy.
    uint64_t hash = util::hash(source);
    std::string source_filename = jitk::hash_filename(compilation_hash, hash, ".c");

    // Make sure all arrays are allocated
    for (bh_base *base: symbols.getParams()) {
        bh_data_malloc(base, true);
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
        assert(base->getDataPtr() != nullptr);
        data_list.push_back(base->getDataPtr());
    }

    // And the offset-and-strides
    vector<uint64_t> offset_and_strides;
    offset_and_strides.reserve(symbols.offsetStrideViews().size());
    for (const bh_view *view: symbols.offsetStrideViews()) {
        const auto t = (uint64_t) view->start;
        offset_and_strides.push_back(t);
        for (int i = 0; i < view->ndim; ++i) {
            const auto s = (uint64_t) view->stride[i];
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
void EnginePGAS::loopHeadWriter(const jitk::SymbolTable &symbols,
                                jitk::Scope &scope,
                                const jitk::LoopB &block,
                                const vector<uint64_t> &thread_stack,
                                stringstream &out) {
    int64_t local_size = block.size;
    int64_t local_offset = 0;

    if (block.rank == 0) {
        // Get the number of processes
        int world_size;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);

        // Get the rank of the process
        int world_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

        if (block._sweeps.empty()) {
            local_size = block.size / world_size;
            local_offset = world_rank * local_size;
            if (world_rank == world_size - 1) {
                local_size += block.size % world_size;
            }
        } else if (world_rank != 0) {
            local_size = 0; // Only rank zero will execute the outermost loop if sweeped
        }
    }

    // Write the for-loop header
    string itername;
    {
        stringstream t;
        t << "i" << block.rank;
        itername = t.str();
    }
    out << "for(uint64_t " << itername << " = " << local_offset << "; ";
    out << itername << " < " << local_offset + local_size << "; ++" << itername << ") {\n";
}

void EnginePGAS::writeKernel(const LoopB &kernel,
                             const jitk::SymbolTable &symbols,
                             const std::vector<bh_base *> &kernel_temps,
                             uint64_t codegen_hash,
                             std::stringstream &ss) {

    assert(kernel.rank == -1);

    // Write the need includes
    ss << "#include <stdint.h>\n";
    ss << "#include <stdlib.h>\n";
    ss << "#include <stdbool.h>\n";
    ss << "#include <complex.h>\n";
    ss << "#include <tgmath.h>\n";
    ss << "#include <math.h>\n";
    ss << "#include <stdio.h>\n";
    ss << "#include <assert.h>\n";
    if (symbols.useRandom()) { // Write the random function
        ss << "#include <kernel_dependencies/random123_openmp.h>\n";
    }
    writeUnionType(ss); // We always need to declare the union of all constant data types
    ss << "\n";

    ss << "#include <mpi.h>\n";
    ss << "// Get the number of processes\n"
          "int world_size;\n"
          "// Get the rank of the process\n"
          "int world_rank;\n\n";
    ss << "\n";

    // Write the header of the execute function
    ss << "void execute_" << codegen_hash;
    writeKernelFunctionArguments(symbols, ss, nullptr);

    // Write the block that makes up the body of 'execute()'
    ss << "{\n";

    ss << "    MPI_Comm_size(MPI_COMM_WORLD, &world_size);\n"
          "    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);\n";
    ss << "    MPI_Barrier(MPI_COMM_WORLD);\n";

    for (bh_base *base: symbols.getParams()) {
        ss << "    MPI_Win win" << symbols.baseID(base) << ";\n";
        ss << "    MPI_Win_create(a" << symbols.baseID(base) << ", "
           << base->pgas.localSize() * bh_type_size(base->dtype())
           << ", " << bh_type_size(base->dtype()) << ", MPI_INFO_NULL, MPI_COMM_WORLD, &win" << symbols.baseID(base)
           << ");\n";
    }

    // Write allocations of the kernel temporaries
    for (const bh_base *b: kernel_temps) {
        util::spaces(ss, 4);
        ss << writeType(b->dtype()) << " * __restrict__ a" << symbols.baseID(b) << " = malloc(" << b->nbytes()
           << ");\n";
    }
    ss << "\n";

    writeBlock(symbols, nullptr, kernel, {}, false, ss);

    ss << "    MPI_Barrier(MPI_COMM_WORLD);\n";

    // Write frees of the kernel temporaries
    ss << "\n";
    for (const bh_base *b: kernel_temps) {
        util::spaces(ss, 4);
        ss << "free(" << "a" << symbols.baseID(b) << ");\n";
    }

    for (bh_base *base: symbols.getParams()) {
        ss << "    MPI_Win_free(&win" << symbols.baseID(base) << ");\n";
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
            ss << writeType(b->dtype()) << " *a" << symbols.baseID(b);
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


void write_mpi_indexes(const Scope &scope, int rank, const bh_view &view, std::stringstream &out) {
    util::spaces(out, 12 + rank * 4);
    out << "int64_t global_idx = ";
    write_array_index(scope, view, out);
    out << ";\n";
    util::spaces(out, 12 + rank * 4);
    out << "int64_t local_size = " << view.base->nelem() << " / world_size;\n";
    util::spaces(out, 12 + rank * 4);
    out << "int owner_rank = (local_size>0)?(global_idx/local_size):(world_size-1);\n";
    util::spaces(out, 12 + rank * 4);
    out << "int64_t owner_offset = (local_size>0)?(global_idx%local_size):(global_idx);\n";
    util::spaces(out, 12 + rank * 4);
    out << "if (local_size > 0 && owner_rank >= world_size) {\n";
    util::spaces(out, 16 + rank * 4);
    out << "owner_rank = world_size-1;\n";
    util::spaces(out, 16 + rank * 4);
    out << "owner_offset = global_idx - local_size * (world_size-1);\n";
    util::spaces(out, 12 + rank * 4);
    out << "}\n";
}

void
write_mpi_get(const SymbolTable &symbols, const Scope &scope, int rank, const bh_view &view, std::stringstream &out,
              bool always_mpi = false) {
    util::spaces(out, 8 + rank * 4);
    out << "{ // PGAS Get \n";
    write_mpi_indexes(scope, rank, view, out);
    util::spaces(out, 12 + rank * 4);
    if (always_mpi) {
        out << "{\n";
    } else {
        out << "if (owner_rank != world_rank) {\n";
    }
    util::spaces(out, 16 + rank * 4);
    out << "MPI_Win_lock (MPI_LOCK_SHARED, owner_rank, 0, win" << symbols.baseID(view.base) << ");\n";
    util::spaces(out, 16 + rank * 4);
    out << "MPI_Get(&" << scope.getName(view) << ", " << bh_type_size(view.base->dtype())
        << ", MPI_BYTE, owner_rank, owner_offset, " << bh_type_size(view.base->dtype())
        << ", MPI_BYTE, win" << symbols.baseID(view.base) << ");\n";
    util::spaces(out, 16 + rank * 4);
    out << "MPI_Win_unlock (owner_rank, win" << symbols.baseID(view.base) << ");\n";
    util::spaces(out, 16 + rank * 4);
    out
            << "printf(\"%d: MPI_Get(%ld) - owner_rank: %d, owner_offset: %ld, local_size: %ld, value: %ld\\n\", world_rank, global_idx, owner_rank, owner_offset, local_size, "
            << scope.getName(view) << ");\n";
    util::spaces(out, 12 + rank * 4);
    out << "}";
    if (always_mpi) {
        out << "\n";
    } else {
        out << "else {\n";
        util::spaces(out, 16 + rank * 4);
        out << scope.getName(view) << " = a" << symbols.baseID(view.base) << "[owner_offset];\n";
        util::spaces(out, 16 + rank * 4);
        out
                << "printf(\"%d: Local_Get(%ld) - owner_rank: %d, owner_offset: %ld, local_size: %ld, value: %d\\n\", world_rank, global_idx, owner_rank, owner_offset, local_size, (int)"
                << scope.getName(view) << ");\n";
        util::spaces(out, 12 + rank * 4);
        out << "}\n";
    }
    util::spaces(out, 8 + rank * 4);
    out << "}\n";
}

void
write_mpi_put(const SymbolTable &symbols, const Scope &scope, int rank, const bh_view &view, std::stringstream &out,
              bool always_mpi = false) {
    util::spaces(out, 8 + rank * 4);
    out << "{ // PGAS Put \n";
    write_mpi_indexes(scope, rank, view, out);
    util::spaces(out, 12 + rank * 4);
    if (always_mpi) {
        out << "{\n";
    } else {
        out << "if (owner_rank != world_rank) {\n";
    }
    util::spaces(out, 16 + rank * 4);
    out
            << "printf(\"%d: MPI_Put(%ld) - owner_rank: %d, owner_offset: %ld, local_size: %ld, value: %d\\n\", world_rank, global_idx, owner_rank, owner_offset, local_size, (int)"
            << scope.getName(view) << ");\n";
    util::spaces(out, 16 + rank * 4);
    out << "MPI_Win_lock(MPI_LOCK_SHARED, owner_rank, 0, win" << symbols.baseID(view.base) << ");\n";
    util::spaces(out, 16 + rank * 4);
    out << "MPI_Put(&" << scope.getName(view) << ", " << bh_type_size(view.base->dtype())
        << ", MPI_BYTE, owner_rank, owner_offset, " << bh_type_size(view.base->dtype())
        << ", MPI_BYTE, win" << symbols.baseID(view.base) << ");\n";
    util::spaces(out, 16 + rank * 4);
    out << "MPI_Win_unlock (owner_rank, win" << symbols.baseID(view.base) << ");\n";
    util::spaces(out, 12 + rank * 4);
    out << "}";
    if (always_mpi) {
        out << "\n";
    } else {
        out << "else {\n";
        util::spaces(out, 16 + rank * 4);
        out
                << "printf(\"%d: Local_Put(%ld) - owner_rank: %d, owner_offset: %ld, local_size: %ld, value: %d\\n\", world_rank, global_idx, owner_rank, owner_offset, local_size, (int)"
                << scope.getName(view) << ");\n";
        util::spaces(out, 16 + rank * 4);
        out << "a" << symbols.baseID(view.base) << "[owner_offset] = ";
        scope.getName(view, out);
        out << ";\n";
        util::spaces(out, 12 + rank * 4);
        out << "}\n";
    }
    util::spaces(out, 8 + rank * 4);
    out << "}\n";
}

void EnginePGAS::writeBlock(const SymbolTable &symbols,
                            const Scope *parent_scope,
                            const LoopB &kernel,
                            const std::vector<uint64_t> &thread_stack,
                            bool opencl,
                            std::stringstream &out) {

    if (kernel.isSystemOnly()) {
        out << "// Removed loop with only system instructions\n";
        return;
    }

    jitk::Scope scope(symbols, parent_scope);

    // Declare temporary arrays
    {
        const set<bh_base *> &local_tmps = kernel.getLocalTemps();
        for (const jitk::InstrPtr &instr: jitk::iterator::allInstr(kernel)) {
            for (const auto &view: instr->getViews()) {
                if (util::exist(local_tmps, view.base)) {
                    if (not(scope.isDeclared(view) or symbols.isAlwaysArray(view.base))) {
                        scope.insertTmp(view.base);
                        util::spaces(out, 8 + kernel.rank * 4);
                        scope.writeDeclaration(view, writeType(view.base->dtype()), out);
                        out << " // tmp\n";
                    }
                }
            }
        }
    }

    // Let's declare indexes if we are not at the kernel level (rank == -1)
    if (kernel.rank >= 0) {
        for (const InstrPtr &instr: jitk::iterator::allLocalInstr(kernel)) {
            for (size_t i = 0; i < instr->operand.size(); ++i) {
                const bh_view &view = instr->operand[i];
                if (symbols.existIdxID(view) and scope.isArray(view)) {
                    if (not scope.isIdxDeclared(view)) {
                        util::spaces(out, 8 + kernel.rank * 4);
                        int hidden_axis = BH_MAXDIM;
                        if (i == 0 and bh_opcode_is_reduction(instr->opcode)) {
                            hidden_axis = instr->sweep_axis();
                        }
                        scope.writeIdxDeclaration(view, writeType(bh_type::UINT64), hidden_axis, out);
                        out << "\n";
                    }
                }
            }
        }
    }

    // Declare scalar replacement of outputs that reduces over the innermost axis in the child block
    {
        for (const jitk::Block &b1: kernel._block_list) {
            if (not b1.isInstr()) {
                for (const InstrPtr &instr: jitk::iterator::allLocalInstr(b1.getLoop())) {
                    if (bh_opcode_is_reduction(instr->opcode) and jitk::sweeping_innermost_axis(instr)) {
                        const bh_view &view = instr->operand[0];
                        if (not(scope.isDeclared(view) or symbols.isAlwaysArray(view.base))) {
                            scope.insertScalarReplaced(view);
                            util::spaces(out, 8 + kernel.rank * 4);
                            scope.writeDeclaration(view, writeType(view.base->dtype()), out);
                            out << "// For reductions inner-most";
                            out << "\n";
                        }
                    }
                }
            }
        }
    }

    // Declare scalar-replacement of all
    {
        for (const InstrPtr &instr: jitk::iterator::allLocalInstr(kernel)) {
            for (size_t i = 0; i < instr->operand.size(); ++i) {
                const bh_view &view = instr->operand[i];
                if (not(view.isConstant() or scope.isDeclared(view) or symbols.isAlwaysArray(view.base))) {
                    scope.insertScalarReplaced(view);
                    util::spaces(out, 8 + kernel.rank * 4);
                    scope.writeDeclaration(view, writeType(view.base->dtype()), out);
                    out << "\n";
                    // Let's load data into the scalar-replaced variable
                    if (not(i == 0 and instr->constructor)) { // No need to load data into a new output
                        write_mpi_get(symbols, scope, kernel.rank, view, out);
                    }
                }
            }
        }
    }

    // Write the for-loop body
    for (const Block &b: kernel._block_list) {
        if (b.isInstr()) { // Finally, let's write the instruction
            if (b.getInstr() != nullptr and not bh_opcode_is_system(b.getInstr()->opcode)) {
                const InstrPtr &instr = b.getInstr();
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
                writeInstr(scope, *instr, 4 + b.rank() * 4, opencl, out);
            }
        } else {
            util::spaces(out, 4 + b.rank() * 4);
            loopHeadWriter(symbols, scope, b.getLoop(), thread_stack, out);
            writeBlock(symbols, &scope, b.getLoop(), thread_stack, opencl, out);
            util::spaces(out, 4 + b.rank() * 4);
            out << "}\n";
        }
    }

    // Let's copy the scalar replaced back to the original array
    for (const InstrPtr &instr: jitk::iterator::allLocalInstr(kernel)) {
        if (not instr->operand.empty()) {
            const bh_view &view = instr->operand[0];
            if (scope.isScalarReplaced(view)) {
                write_mpi_put(symbols, scope, kernel.rank, view, out);
                scope.eraseScalarReplaced(view);
            }
        }
    }
}

std::string EnginePGAS::info() const {
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
const std::string EnginePGAS::writeType(bh_type dtype) {
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
