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
#pragma once

#include <bh_config_parser.hpp>
#include <jitk/statistics.hpp>

#include <bh_view.hpp>
#include <bh_component.hpp>
#include <bh_instruction.hpp>
#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;

namespace bohrium {
namespace jitk {

class Engine {
  public:
    const ConfigParser &config;
    Statistics &stat;

    FuseCache fcache;

    CodegenCache codegen_cache;

    const bool verbose;

    // Maximum number of cache files
    const int64_t cache_file_max;

    // Path to a temporary directory for the source and object files
    const boost::filesystem::path tmp_dir;

    // Path to the temporary directory of the source files
    const boost::filesystem::path tmp_src_dir;

    // Path to the temporary directory of the binary files (e.g. .so files)
    const boost::filesystem::path tmp_bin_dir;

    // Path to the directory of the cached binary files (e.g. .so files)
    const boost::filesystem::path cache_bin_dir;

    // The hash of the JIT compilation command
    size_t compilation_hash;

    Engine(const ConfigParser &config, Statistics &stat) :
      config(config),
      stat(stat),
      fcache(stat),
      codegen_cache(stat),
      verbose(config.defaultGet<bool>("verbose", false)),
      cache_file_max(config.defaultGet<int64_t>("cache_file_max", 50000)),
      tmp_dir(get_tmp_path(config)),
      tmp_src_dir(tmp_dir / "src"),
      tmp_bin_dir(tmp_dir / "obj"),
      cache_bin_dir(config.defaultGet<fs::path>("cache_dir", "")),
      compilation_hash(0) {
        // Let's make sure that the directories exist
        jitk::create_directories(tmp_src_dir);
        jitk::create_directories(tmp_bin_dir);

        if (not cache_bin_dir.empty()) {
            jitk::create_directories(cache_bin_dir);
        }
    }

    virtual ~Engine() {}

    virtual void execute(const std::string &source,
                         const std::vector<bh_base*> &non_temps,
                         const std::vector<const bh_view*> &offset_strides,
                         const std::vector<const bh_instruction*> &constants) {
        throw std::runtime_error("execute1 - Abstract method must be implemented!");
    }

    virtual void execute(const std::string &source,
                         const std::vector<bh_base*> &non_temps,
                         const std::vector<const LoopB*> &threaded_blocks,
                         const std::vector<const bh_view*> &offset_strides,
                         const std::vector<const bh_instruction*> &constants) {
        throw std::runtime_error("execute2 - Abstract method must be implemented!");
    }

    virtual void copyToHost(const std::vector<bh_base*> &bases) {
        throw std::runtime_error("copyToHost - Abstract method must be implemented!");
    }
    virtual void copyToHost(const std::set<bh_base*> &bases) {
        throw std::runtime_error("copyToHost - Abstract method must be implemented!");
    }

    virtual void copyToDevice(const std::vector<bh_base*> &base_list) {
        throw std::runtime_error("copyToDevice - Abstract method must be implemented!");
    }
    virtual void copyToDevice(const std::set<bh_base*> &base_list) {
        throw std::runtime_error("copyToDevice - Abstract method must be implemented!");
    }

    virtual void set_constructor_flag(std::vector<bh_instruction*> &instr_list) = 0;

    virtual void delBuffer(bh_base* &base) {
        throw std::runtime_error("delBuffer - Abstract method must be implemented!");
    }

    // TODO: Better union of these two `handle_execution` methods
    virtual void handle_execution(BhIR *bhir) {
        throw std::runtime_error("handle_execution - Abstract method must be implemented!");
    }
    void handle_execution(component::ComponentImplWithChild &comp, BhIR *bhir) {
        using namespace std;

        const auto texecution = chrono::steady_clock::now();

        map<string, bool> kernel_config = {
            { "verbose",        config.defaultGet<bool>("verbose",       false) },
            { "strides_as_var", config.defaultGet<bool>("strides_as_var", true) },
            { "index_as_var",   config.defaultGet<bool>("index_as_var",   true) },
            { "const_as_var",   config.defaultGet<bool>("const_as_var",   true) },
            { "use_volatile",   config.defaultGet<bool>("use_volatile",  false) }
        };
        const uint64_t parallel_threshold = config.defaultGet<uint64_t>("parallel_threshold", 1000);

        // Some statistics
        stat.record(*bhir);

        // Let's start by cleanup the instructions from the 'bhir'
        vector<bh_instruction*> instr_list;

        set<bh_base*> frees;
        instr_list = jitk::remove_non_computed_system_instr(bhir->instr_list, frees);

        // Let's free device buffers and array memory
        for(bh_base *base: frees) {
            delBuffer(base);
            bh_data_free(base);
        }

        // Set the constructor flag
        if (config.defaultGet<bool>("array_contraction", true)) {
            set_constructor_flag(instr_list);
        } else {
            for (bh_instruction *instr: instr_list) {
                instr->constructor = false;
            }
        }

        // Let's get the block list
        // NB: 'avoid_rank0_sweep' is set to true when we have a child to offload to.
        const vector<jitk::Block> block_list = get_block_list(instr_list, config, fcache, stat, &(comp.child) != nullptr);

        for(const jitk::Block &block: block_list) {
            assert(not block.isInstr());

            // Let's create the symbol table for the kernel
            const jitk::SymbolTable symbols(block.getAllInstr(), block.getLoop().getAllNonTemps(), kernel_config);
            stat.record(symbols);

            // We can skip a lot of steps if the kernel does no computation
            const bool kernel_is_computing = not block.isSystemOnly();

            // Find the parallel blocks
            const vector<const jitk::LoopB*> threaded_blocks = find_threaded_blocks(block, stat, parallel_threshold);

            // We might have to offload the execution to the CPU
            if (threaded_blocks.size() == 0 and kernel_is_computing) {
                if (kernel_config["verbose"]) {
                    cout << "Offloading to CPU\n";
                }

                if (&(comp.child) == nullptr) {
                    throw runtime_error("handle_execution(): threaded_blocks cannot be empty when child == NULL!");
                }

                auto toffload = chrono::steady_clock::now();

                // Let's copy all non-temporary to the host
                copyToHost(symbols.getParams());

                // Let's free device buffers
                for (bh_base *base: symbols.getFrees()) {
                    delBuffer(base);
                }

                // Let's send the kernel instructions to our child
                vector<bh_instruction> child_instr_list;
                for (const jitk::InstrPtr &instr: block.getAllInstr()) {
                    child_instr_list.push_back(*instr);
                }
                BhIR tmp_bhir(std::move(child_instr_list), bhir->getSyncs());
                comp.child.execute(&tmp_bhir);
                stat.time_offload += chrono::steady_clock::now() - toffload;
                continue;
            }

            // Let's execute the kernel
            if (kernel_is_computing) {
                // We need a memory buffer on the device for each non-temporary array in the kernel
                copyToDevice(symbols.getParams());

                // Create the constant vector
                vector<const bh_instruction*> constants;
                constants.reserve(symbols.constIDs().size());
                for (const jitk::InstrPtr &instr: symbols.constIDs()) {
                    constants.push_back(&(*instr));
                }

                const auto lookup = codegen_cache.get({ block }, symbols);
                if(lookup.second) {
                    // In debug mode, we check that the cached source code is correct
                    #ifndef NDEBUG
                        stringstream ss;
                        write_kernel(block, symbols, threaded_blocks, ss);
                        if (ss.str().compare(lookup.first) != 0) {
                            cout << "\nCached source code: \n" << lookup.first;
                            cout << "\nReal source code: \n" << ss.str();
                            assert(1 == 2);
                        }
                    #endif
                    execute(lookup.first, symbols.getParams(), threaded_blocks, symbols.offsetStrideViews(), constants);
                } else {
                    const auto tcodegen = chrono::steady_clock::now();
                    stringstream ss;
                    write_kernel(block, symbols, threaded_blocks, ss);
                    string source = ss.str();
                    stat.time_codegen += chrono::steady_clock::now() - tcodegen;
                    execute(source, symbols.getParams(), threaded_blocks, symbols.offsetStrideViews(), constants);
                    codegen_cache.insert(std::move(source), { block }, symbols);
                }
            }

            // Let's copy sync'ed arrays back to the host
            copyToHost(bhir->getSyncs());

            // Let's free device buffers
            for(bh_base *base: symbols.getFrees()) {
                delBuffer(base);
            }

            // Finally, let's cleanup
            for(bh_base *base: symbols.getFrees()) {
                bh_data_free(base);
            }
        }
        stat.time_total_execution += chrono::steady_clock::now() - texecution;
    }

    void write_kernel_function_arguments(const jitk::SymbolTable &symbols,
                                         std::stringstream &ss,
                                         const char *array_type_prefix) {
        // We create the comma separated list of args and saves it in `stmp`
        std::stringstream stmp;
        for (size_t i = 0; i < symbols.getParams().size(); ++i) {
            bh_base *b = symbols.getParams()[i];
            if (array_type_prefix != nullptr) {
                stmp << array_type_prefix << " ";
            }
            stmp << write_type(b->type) << "* __restrict__ a" << symbols.baseID(b) << ", ";
        }

        for (const bh_view *view: symbols.offsetStrideViews()) {
            stmp << write_type(bh_type::UINT64);
            stmp << " vo" << symbols.offsetStridesID(*view) << ", ";
            for (int i = 0; i < view->ndim; ++i) {
                stmp << write_type(bh_type::UINT64) << " vs" << symbols.offsetStridesID(*view) << "_" << i << ", ";
            }
        }

        if (not symbols.constIDs().empty()) {
            for (auto it = symbols.constIDs().begin(); it != symbols.constIDs().end(); ++it) {
                const InstrPtr &instr = *it;
                stmp << "const " << write_type(instr->constant.type) << " c" << symbols.constID(*instr) << ", ";
            }
        }

        // And then we write `stmp` into `ss` excluding the last comma
        const std::string strtmp = stmp.str();
        if (strtmp.empty()) {
            ss << "()";
        } else {
            // Excluding the last comma
            ss << "(" << strtmp.substr(0, strtmp.size()-2) << ")";
        }
    }

    // Writes a loop block, which corresponds to a parallel for-loop.
    // The two functions 'type_writer' and 'head_writer' should write the
    // backend specific data type names and for-loop headers respectively.
    void write_loop_block(const jitk::SymbolTable &symbols,
                          const jitk::Scope *parent_scope,
                          const jitk::LoopB &block,
                          const std::vector<const jitk::LoopB*> &threaded_blocks,
                          bool opencl,
                          std::stringstream &out) {
        using namespace std;

        if (block.isSystemOnly()) {
            out << "// Removed loop with only system instructions\n";
            return;
        }

        util::spaces(out, 4 + block.rank*4);

        // Order all sweep instructions by the viewID of their first operand.
        // This makes the source of the kernels more identical, which improve the code and compile caches.
        const vector<jitk::InstrPtr> ordered_block_sweeps = order_sweep_set(block._sweeps, symbols);

        // Let's find the local temporary arrays and the arrays to scalar replace
        const set<bh_base *> &local_tmps = block.getLocalTemps();

        // Let's scalar replace reduction outputs that reduces over the innermost axis
        vector<const bh_view*> scalar_replaced_reduction_outputs;
        for (const InstrPtr &instr: ordered_block_sweeps) {
            if (bh_opcode_is_reduction(instr->opcode) and sweeping_innermost_axis(instr)) {
                if (local_tmps.find(instr->operand[0].base) == local_tmps.end() and
                        (parent_scope == nullptr or parent_scope->isArray(instr->operand[0]))) {
                    scalar_replaced_reduction_outputs.push_back(&instr->operand[0]);
                }
            }
        }

        // Let's scalar replace input-only arrays that are used multiple times
        vector<const bh_view*> srio = scalar_replaced_input_only(block, parent_scope, local_tmps);

        // And then create the scope
        jitk::Scope scope(symbols, parent_scope, local_tmps, scalar_replaced_reduction_outputs, srio);

        // When a reduction output is a scalar (e.g. because of array contraction or scalar replacement),
        // it should be declared before the for-loop
        for (const InstrPtr &instr: ordered_block_sweeps) {
            if (bh_opcode_is_reduction(instr->opcode)) {
                const bh_view &output = instr->operand[0];
                if (not scope.isDeclared(output) and not scope.isArray(output)) {
                    // Let's write the declaration of the scalar variable
                    scope.writeDeclaration(output, write_type(output.base->type), out);
                    out << "\n";
                    util::spaces(out, 4 + block.rank * 4);
                }
            }
        }

        // Find indexes we will declare later
        vector<const bh_view*> indexes = get_indexes(block, scope, symbols);

        // We might not have to loop "peel" if all reduction have an identity value and writes to a scalar
        bool peel = need_to_peel(ordered_block_sweeps, scope);

        // When not peeling, we need a neutral initial reduction value
        if (not peel) {
            for (const jitk::InstrPtr &instr: ordered_block_sweeps) {
                const bh_view &view = instr->operand[0];
                if (not scope.isArray(view) and not scope.isDeclared(view)) {
                    scope.writeDeclaration(view, write_type(view.base->type), out);
                    out << "\n";
                    util::spaces(out, 4 + block.rank * 4);
                }
                scope.getName(view, out);
                out << " = ";
                write_reduce_identity(instr->opcode, view.base->type, out);
                out << ";\n";
                util::spaces(out, 4 + block.rank * 4);
            }
        }

        // If this block is sweeped, we will "peel" the for-loop such that the
        // sweep instruction is replaced with BH_IDENTITY in the first iteration
        if (block._sweeps.size() > 0 and peel) {
            jitk::Scope peeled_scope(scope);
            jitk::LoopB peeled_block(block);
            for (const InstrPtr instr: ordered_block_sweeps) {
                // The input is the same as in the sweep
                bh_instruction sweep_instr(BH_IDENTITY, {instr->operand[0], instr->operand[1]});

                // But the output needs an extra dimension when we are reducing to a non-scalar
                if (bh_opcode_is_reduction(instr->opcode) and instr->operand[1].ndim > 1) {
                    sweep_instr.operand[0].insert_axis(instr->constant.get_int64(), 1, 0);
                }
                peeled_block.replaceInstr(instr, sweep_instr);
            }
            string itername;
            { stringstream t; t << "i" << block.rank; itername = t.str(); }
            out << "{ // Peeled loop, 1. sweep iteration\n";
            util::spaces(out, 8 + block.rank*4);
            out << write_type(bh_type::UINT64) << " " << itername << " = 0;\n";

            // Write temporary and scalar replaced array declarations
            for (const InstrPtr &instr: block.getLocalInstr()) {
                for (const bh_view *view: instr->get_views()) {
                    if (not peeled_scope.isDeclared(*view)) {
                        if (peeled_scope.isTmp(view->base)) {
                            util::spaces(out, 8 + block.rank * 4);
                            peeled_scope.writeDeclaration(*view, write_type(view->base->type), out);
                            out << "\n";
                        } else if (peeled_scope.isScalarReplaced_R(*view)) {
                            util::spaces(out, 8 + block.rank * 4);
                            peeled_scope.writeDeclaration(*view, write_type(view->base->type), out);
                            out << " " << peeled_scope.getName(*view) << " = a" << symbols.baseID(view->base);
                            write_array_subscription(peeled_scope, *view, out);
                            out << ";";
                            out << "\n";
                        }
                    }
                }
            }
            // Write the indexes declarations
            for (const bh_view *view: indexes) {
                if (not peeled_scope.isIdxDeclared(*view)) {
                    util::spaces(out, 8 + block.rank * 4);
                    peeled_scope.writeIdxDeclaration(*view, write_type(bh_type::UINT64), out);
                    out << "\n";
                }
            }
            out << "\n";
            for (const Block &b: peeled_block._block_list) {
                if (b.isInstr()) {
                    if (b.getInstr() != nullptr and not bh_opcode_is_system(b.getInstr()->opcode)) {
                        util::spaces(out, 4 + b.rank()*4);
                        write_instr(peeled_scope, *b.getInstr(), out, opencl);
                    }
                } else {
                    write_loop_block(symbols, &peeled_scope, b.getLoop(), threaded_blocks, opencl, out);
                }
            }
            util::spaces(out, 4 + block.rank*4);
            out << "}\n";
            util::spaces(out, 4 + block.rank*4);
        }

        // Write the for-loop header
        loop_head_writer(symbols, scope, block, peel, threaded_blocks, out);

        // Write temporary and scalar replaced array declarations
        for (const InstrPtr &instr: block.getLocalInstr()) {
            for (const bh_view *view: instr->get_views()) {
                if (not scope.isDeclared(*view)) {
                    if (scope.isTmp(view->base)) {
                        util::spaces(out, 8 + block.rank * 4);
                        scope.writeDeclaration(*view, write_type(view->base->type), out);
                        out << "\n";
                    } else if (scope.isScalarReplaced_R(*view)) {
                        util::spaces(out, 8 + block.rank * 4);
                        scope.writeDeclaration(*view, write_type(view->base->type), out);
                        out << " " << scope.getName(*view) << " = a" << symbols.baseID(view->base);
                        write_array_subscription(scope, *view, out);
                        out << ";";
                        out << "\n";
                    }
                }
            }
        }
        // Write the indexes declarations
        for (const bh_view *view: indexes) {
            if (not scope.isIdxDeclared(*view)) {
                util::spaces(out, 8 + block.rank * 4);
                scope.writeIdxDeclaration(*view, write_type(bh_type::UINT64), out);
                out << "\n";
            }
        }

        // Write the for-loop body
        // The body in OpenCL and OpenMP are very similar but OpenMP might need to insert "#pragma omp atomic/critical"
        if (opencl) {
            for (const Block &b: block._block_list) {
                if (b.isInstr()) { // Finally, let's write the instruction
                    if (b.getInstr() != NULL and not bh_opcode_is_system(b.getInstr()->opcode)) {
                        util::spaces(out, 4 + b.rank()*4);
                        write_instr(scope, *b.getInstr(), out, true);
                    }
                } else {
                    write_loop_block(symbols, &scope, b.getLoop(), threaded_blocks, opencl, out);
                }
            }
        } else {
            for (const Block &b: block._block_list) {
                if (b.isInstr()) { // Finally, let's write the instruction
                    const InstrPtr instr = b.getInstr();
                    if (not bh_opcode_is_system(instr->opcode)) {
                        if (instr->operand.size() > 0) {
                            if (scope.isOpenmpAtomic(instr->operand[0])) {
                                util::spaces(out, 4 + b.rank()*4);
                                out << "#pragma omp atomic\n";
                            } else if (scope.isOpenmpCritical(instr->operand[0])) {
                                util::spaces(out, 4 + b.rank()*4);
                                out << "#pragma omp critical\n";
                            }
                        }
                        util::spaces(out, 4 + b.rank()*4);
                        write_instr(scope, *instr, out);
                    }
                } else {
                    write_loop_block(symbols, &scope, b.getLoop(), threaded_blocks, opencl, out);
                }
            }
        }
        util::spaces(out, 4 + block.rank*4);
        out << "}\n";

        // Let's copy the scalar replaced reduction outputs back to the original array
        for (const bh_view *view: scalar_replaced_reduction_outputs) {
            util::spaces(out, 4 + block.rank*4);
            out << "a" << symbols.baseID(view->base);
            write_array_subscription(scope, *view, out, true);
            out << " = ";
            scope.getName(*view, out);
            out << ";\n";
        }
    }

    // TODO: Better union of these two `write_kernel` methods
    virtual void write_kernel(const std::vector<Block> &block_list,
                              const SymbolTable &symbols,
                              const std::vector<bh_base*> &kernel_temps,
                              std::stringstream &ss) {
        throw std::runtime_error("write_kernel1 - Abstract method must be implemented!");
    };

    virtual void write_kernel(const Block &block,
                              const SymbolTable &symbols,
                              const std::vector<const LoopB*> &threaded_blocks,
                              std::stringstream &ss) {
        throw std::runtime_error("write_kernel2 - Abstract method must be implemented!");
    };

    virtual void loop_head_writer(const SymbolTable &symbols,
                                  Scope &scope,
                                  const LoopB &block,
                                  bool loop_is_peeled,
                                  const std::vector<const LoopB *> &threaded_blocks,
                                  std::stringstream &out) = 0;

    virtual std::string info() const = 0;

    virtual const std::string write_type(bh_type dtype) = 0;

    // TODO: Unify the two `handle_extmethod`. They seem very alike.
    template <typename T>
    void handle_extmethod(T &comp, BhIR *bhir) {
        std::vector<bh_instruction> instr_list;

        for (bh_instruction &instr: bhir->instr_list) {
            auto ext = comp.extmethods.find(instr.opcode);

            if (ext != comp.extmethods.end()) { // Execute the instructions up until now
                BhIR b(std::move(instr_list), bhir->getSyncs());
                comp.execute(&b);
                instr_list.clear(); // Notice, it is legal to clear a moved vector.
                const auto texecution = std::chrono::steady_clock::now();
                ext->second.execute(&instr, nullptr); // Execute the extension method
                stat.time_ext_method += std::chrono::steady_clock::now() - texecution;
            } else {
                instr_list.push_back(instr);
            }
        }

        bhir->instr_list = instr_list;
    }

    template <typename T>
    void handle_extmethod(T &comp, BhIR *bhir, std::set<bh_opcode> child_extmethods) {
        std::vector<bh_instruction> instr_list;

        for (bh_instruction &instr: bhir->instr_list) {
            auto ext = comp.extmethods.find(instr.opcode);
            auto childext = child_extmethods.find(instr.opcode);

            if (ext != comp.extmethods.end() or childext != child_extmethods.end()) {
                // Execute the instructions up until now
                BhIR b(std::move(instr_list), bhir->getSyncs());
                comp.execute(&b);
                instr_list.clear(); // Notice, it is legal to clear a moved vector.

                if (ext != comp.extmethods.end()) {
                    const auto texecution = std::chrono::steady_clock::now();
                    ext->second.execute(&instr, &*this); // Execute the extension method
                    stat.time_ext_method += std::chrono::steady_clock::now() - texecution;
                } else if (childext != child_extmethods.end()) {
                    // We let the child component execute the instruction
                    std::set<bh_base *> ext_bases = instr.get_bases();

                    copyToHost(ext_bases);

                    std::vector<bh_instruction> child_instr_list;
                    child_instr_list.push_back(instr);
                    b.instr_list = child_instr_list;
                    comp.child.execute(&b);
                }
            } else {
                instr_list.push_back(instr);
            }
        }

        bhir->instr_list = instr_list;
    }

  private:
    bool need_to_peel(const std::vector<InstrPtr> &ordered_block_sweeps, const Scope &scope) {
        for (const InstrPtr &instr: ordered_block_sweeps) {
            const bh_view &v = instr->operand[0];
            if (not (has_reduce_identity(instr->opcode) and (scope.isScalarReplaced(v) or scope.isTmp(v.base)))) {
                return true;
            }
        }
        return false;
    }

    std::vector<const bh_view*> get_indexes(const LoopB &block, const Scope &scope, const SymbolTable &symbols) {
        std::vector<const bh_view*> indexes;
        std::set<bh_view, idx_less> candidates;
        for (const InstrPtr &instr: block.getLocalInstr()) {
            for (const bh_view* view: instr->get_views()) {
                if (symbols.existIdxID(*view) and scope.isArray(*view)) {
                    if (util::exist(candidates, *view)) { // 'view' is used multiple times
                        indexes.push_back(view);
                    } else {
                        candidates.insert(*view);
                    }
                }
            }
        }
        return indexes;
    }
};

}} // namespace
