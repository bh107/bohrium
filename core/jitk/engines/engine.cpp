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

#include <jitk/engines/engine.hpp>

using namespace std;

namespace bohrium {
namespace jitk {

void Engine::writeKernelFunctionArguments(const jitk::SymbolTable &symbols,
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
        ss << "(" << strtmp.substr(0, strtmp.size() - 2) << ")";
    }
}

void Engine::writeBlock(const SymbolTable &symbols,
                        const Scope *parent_scope,
                        const LoopB &kernel,
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
    const vector <jitk::InstrPtr> ordered_block_sweeps = order_sweep_set(sweeps_in_child, symbols);

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
    vector<const bh_view *> scalar_replaced_to_write_back;
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

    // Write the for-loop body
    // The body in OpenCL and OpenMP are very similar but OpenMP might need to insert "#pragma omp atomic/critical"
    if (opencl) {
        for (const Block &b: kernel._block_list) {
            if (b.isInstr()) { // Finally, let's write the instruction
                if (b.getInstr() != nullptr and not bh_opcode_is_system(b.getInstr()->opcode)) {
                    util::spaces(out, 4 + b.rank() * 4);
                    write_instr(scope, *b.getInstr(), out, true);
                }
            } else {
                util::spaces(out, 4 + b.rank() * 4);
                loopHeadWriter(symbols, scope, b.getLoop(), thread_stack, out);
                writeBlock(symbols, &scope, b.getLoop(), thread_stack, opencl, out);
                util::spaces(out, 4 + b.rank() * 4);
                out << "}\n";
            }
        }
    } else {
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
                loopHeadWriter(symbols, scope, b.getLoop(), thread_stack, out);
                writeBlock(symbols, &scope, b.getLoop(), thread_stack, opencl, out);
                util::spaces(out, 4 + b.rank() * 4);
                out << "}\n";
            }
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

void Engine::setConstructorFlag(std::vector<bh_instruction *> &instr_list, std::set<bh_base *> &constructed_arrays) {
    for (bh_instruction *instr: instr_list) {
        instr->constructor = false;
        for (size_t o = 0; o < instr->operand.size(); ++o) {
            const bh_view &v = instr->operand[o];
            if (not bh_is_constant(&v)) {
                if (o == 0 and not util::exist_nconst(constructed_arrays, v.base)) {
                    instr->constructor = true;
                }
                constructed_arrays.insert(v.base);
            }
        }
    }
}

}
} // namespace
