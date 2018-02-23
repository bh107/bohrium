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

namespace bohrium {
namespace jitk {

class Engine {
protected:
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
    uint64_t compilation_hash;

public:
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
      cache_bin_dir(config.defaultGet<boost::filesystem::path>("cache_dir", "")),
      compilation_hash(0) {
        // Let's make sure that the directories exist
        jitk::create_directories(tmp_src_dir);
        jitk::create_directories(tmp_bin_dir);

        if (not cache_bin_dir.empty()) {
            jitk::create_directories(cache_bin_dir);
        }
    }

    virtual ~Engine() {}

    virtual std::string info() const = 0;
    virtual const std::string writeType(bh_type dtype) = 0;
    virtual void setConstructorFlag(std::vector<bh_instruction*> &instr_list) = 0;

protected:
    void writeKernelFunctionArguments(const jitk::SymbolTable &symbols,
                                      std::stringstream &ss,
                                      const char *array_type_prefix) {
        // We create the comma separated list of args and saves it in `stmp`
        std::stringstream stmp;
        for (size_t i = 0; i < symbols.getParams().size(); ++i) {
            bh_base *b = symbols.getParams()[i];
            if (array_type_prefix != nullptr) {
                stmp << array_type_prefix << " ";
            }
            stmp << writeType(b->type) << "* __restrict__ a" << symbols.baseID(b) << ", ";
        }

        for (const bh_view *view: symbols.offsetStrideViews()) {
            stmp << writeType(bh_type::UINT64);
            stmp << " vo" << symbols.offsetStridesID(*view) << ", ";
            for (int i = 0; i < view->ndim; ++i) {
                stmp << writeType(bh_type::UINT64) << " vs" << symbols.offsetStridesID(*view) << "_" << i << ", ";
            }
        }

        if (not symbols.constIDs().empty()) {
            for (auto it = symbols.constIDs().begin(); it != symbols.constIDs().end(); ++it) {
                const InstrPtr &instr = *it;
                stmp << "const " << writeType(instr->constant.type) << " c" << symbols.constID(*instr) << ", ";
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
    void writeLoopBlock(const jitk::SymbolTable &symbols,
                        const jitk::Scope *parent_scope,
                        const jitk::LoopB &block,
                        const std::vector<uint64_t> &thread_stack,
                        bool opencl,
                        std::stringstream &out) {
        using namespace std;

        if (block.isSystemOnly()) {
            out << "// Removed loop with only system instructions\n";
            return;
        }

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
                    util::spaces(out, 4 + block.rank * 4);
                    scope.writeDeclaration(output, writeType(output.base->type), out);
                    out << "\n";
                }
            }
        }

        // Find indexes we will declare later. Notice, `indexes` might include the identical views
        // hence please check `isIdxDeclared()` to avoid duplicates
        std::vector<const bh_view*> indexes;
        {
            std::vector<const bh_view*> indexes;
            for (const InstrPtr &instr: block.getLocalInstr()) {
                for (const bh_view* view: instr->get_views()) {
                    if (symbols.existIdxID(*view) and scope.isArray(*view)) {
                        indexes.push_back(view);
                    }
                }
            }
        }

        // We might not have to loop "peel" if all reduction have an identity value and writes to a scalar
        bool peel = needToPeel(ordered_block_sweeps, scope);

        // When not peeling, we need a neutral initial reduction value
        if (not peel) {
            for (const jitk::InstrPtr &instr: ordered_block_sweeps) {
                const bh_view &view = instr->operand[0];
                if (not scope.isArray(view) and not scope.isDeclared(view)) {
                    util::spaces(out, 4 + block.rank * 4);
                    scope.writeDeclaration(view, writeType(view.base->type), out);
                    out << "\n";
                }
                util::spaces(out, 4 + block.rank * 4);
                scope.getName(view, out);
                out << " = ";
                write_reduce_identity(instr->opcode, view.base->type, out);
                out << ";\n";
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
            util::spaces(out, 4 + block.rank * 4);
            out << "{ // Peeled loop, 1. sweep iteration\n";
            util::spaces(out, 8 + block.rank*4);
            out << writeType(bh_type::UINT64) << " " << itername << " = 0;\n";

            // Write temporary and scalar replaced array declarations
            for (const InstrPtr &instr: block.getLocalInstr()) {
                for (const bh_view *view: instr->get_views()) {
                    if (not peeled_scope.isDeclared(*view)) {
                        if (peeled_scope.isTmp(view->base)) {
                            util::spaces(out, 8 + block.rank * 4);
                            peeled_scope.writeDeclaration(*view, writeType(view->base->type), out);
                            out << "\n";
                        } else if (peeled_scope.isScalarReplaced_R(*view)) {
                            util::spaces(out, 8 + block.rank * 4);
                            peeled_scope.writeDeclaration(*view, writeType(view->base->type), out);
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
                    peeled_scope.writeIdxDeclaration(*view, writeType(bh_type::UINT64), out);
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
                    writeLoopBlock(symbols, &peeled_scope, b.getLoop(), thread_stack, opencl, out);
                }
            }
            util::spaces(out, 4 + block.rank*4);
            out << "}\n";
        }

        // Write the for-loop header
        util::spaces(out, 4 + block.rank*4);
        loopHeadWriter(symbols, scope, block, peel, thread_stack, out);

        // Write temporary and scalar replaced array declarations
        for (const InstrPtr &instr: block.getLocalInstr()) {
            for (const bh_view *view: instr->get_views()) {
                if (not scope.isDeclared(*view)) {
                    if (scope.isTmp(view->base)) {
                        util::spaces(out, 8 + block.rank * 4);
                        scope.writeDeclaration(*view, writeType(view->base->type), out);
                        out << "\n";
                    } else if (scope.isScalarReplaced_R(*view)) {
                        util::spaces(out, 8 + block.rank * 4);
                        scope.writeDeclaration(*view, writeType(view->base->type), out);
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
                scope.writeIdxDeclaration(*view, writeType(bh_type::UINT64), out);
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
                    writeLoopBlock(symbols, &scope, b.getLoop(), thread_stack, opencl, out);
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
                    writeLoopBlock(symbols, &scope, b.getLoop(), thread_stack, opencl, out);
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

    virtual void loopHeadWriter(const SymbolTable &symbols,
                                Scope &scope,
                                const LoopB &block,
                                bool loop_is_peeled,
                                const std::vector<uint64_t> &thread_stack,
                                std::stringstream &out) = 0;

private:
    bool needToPeel(const std::vector<InstrPtr> &ordered_block_sweeps, const Scope &scope) {
        for (const InstrPtr &instr: ordered_block_sweeps) {
            const bh_view &v = instr->operand[0];
            if (not (has_reduce_identity(instr->opcode) and (scope.isScalarReplaced(v) or scope.isTmp(v.base)))) {
                return true;
            }
        }
        return false;
    }

};

}} // namespace
