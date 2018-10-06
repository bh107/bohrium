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

namespace {
std::vector<const bh_view *> scalar_replaced_input_only(const LoopB &block, const Scope *parent_scope) {
    std::vector<const bh_view *> res;

    // We have to ignore output arrays and arrays that are accumulated
    std::set<bh_base *> ignore_bases;
    for (const InstrPtr &instr: block.getLocalInstr()) {
        if (not instr->operand.empty()) {
            ignore_bases.insert(instr->operand[0].base);
        }
        if (bh_opcode_is_accumulate(instr->opcode)) {
            ignore_bases.insert(instr->operand[1].base);
        }
    }
    // First we add a valid view to the set of 'candidates' and if we encounter the view again
    // we add it to the 'result'
    std::set<bh_view> candidates;
    for (const InstrPtr &instr: block.getLocalInstr()) {
        for (size_t i = 1; i < instr->operand.size(); ++i) {
            const bh_view &input = instr->operand[i];
            if ((not input.isConstant()) and ignore_bases.find(input.base) == ignore_bases.end()) {
                if (parent_scope == nullptr or parent_scope->isArray(input)) {
                    if (util::exist(candidates, input)) { // 'input' is used multiple times
                        res.push_back(&input);
                    } else {
                        candidates.insert(input);
                    }
                }
            }
        }
    }
    return res;
}
}

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

    jitk::Scope scope(symbols, parent_scope);

    // Declare temporary arrays
    {
        const set<bh_base *> &local_tmps = kernel.getLocalTemps();
        for (const jitk::InstrPtr &instr: iterator::allInstr(kernel)) {
            for (const auto &view: instr->getViews()) {
                if (util::exist(local_tmps, view.base)) {
                    if (not (scope.isDeclared(view) or symbols.isAlwaysArray(view.base))) {
                        scope.insertTmp(view.base);
                        util::spaces(out, 8 + kernel.rank * 4);
                        scope.writeDeclaration(view, writeType(view.base->type), out);
                        out << "\n";
                    }
                }
            }
        }
    }

    // Let's declare indexes if we are not at the kernel level (rank == -1)
    if (kernel.rank >= 0) {
        for (const jitk::Block &block: kernel._block_list) {
            if (block.isInstr()) {
                const jitk::InstrPtr &instr = block.getInstr();
                for (const bh_view &view: instr->getViews()) {
                    if (symbols.existIdxID(view) and scope.isArray(view)) {
                        if (not scope.isIdxDeclared(view)) {
                            util::spaces(out, 8 + kernel.rank * 4);
                            scope.writeIdxDeclaration(view, writeType(bh_type::UINT64), out);
                            out << "\n";
                        }
                    }
                }
            }
        }
    }

    // Declare scalar replacement of outputs that reduces over the innermost axis in the child block
    vector<pair<const bh_view *, int> > scalar_replaced_to_write_back; // Pair of the view and hidden_axis
    {
        for (const jitk::Block &b1: kernel._block_list) {
            if (not b1.isInstr()) {
                for (const jitk::Block &b2: b1.getLoop()._block_list) {
                    if (b2.isInstr()) {
                        const InstrPtr &instr = b2.getInstr();
                        if (bh_opcode_is_reduction(instr->opcode) and jitk::sweeping_innermost_axis(instr)) {
                            const bh_view &view = instr->operand[0];
                            if (not(scope.isDeclared(view) or symbols.isAlwaysArray(view.base))) {
                                scope.insertScalarReplaced_RW(view);
                                util::spaces(out, 8 + kernel.rank * 4);
                                scope.writeDeclaration(view, writeType(view.base->type), out);
                                out << "// For reductions";
                                out << "\n";
                                scalar_replaced_to_write_back.emplace_back(&view, instr->sweep_axis());
                            }
                        }
                    }
                }
            }
        }
    }

    // Let's scalar replace input-only arrays that are used multiple times
    {
        for (const bh_view *view: scalar_replaced_input_only(kernel, parent_scope)) {
            if (not(scope.isDeclared(*view) or symbols.isAlwaysArray(view->base))) {
                scope.insertScalarReplaced_R(*view);
                util::spaces(out, 8 + kernel.rank * 4);
                scope.writeDeclaration(*view, writeType(view->base->type), out);
                out << " " << scope.getName(*view) << " = a" << symbols.baseID(view->base);
                write_array_subscription(scope, *view, out, false);
                out << "; // For input-only";
                out << "\n";
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
                const InstrPtr &instr = b.getInstr();
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

    // Let's copy the scalar replaced back to the original array
    for (const auto view_and_hidden_axis: scalar_replaced_to_write_back) {
        const bh_view *view = view_and_hidden_axis.first;
        const int hidden_axis = view_and_hidden_axis.second;
        util::spaces(out, 8 + kernel.rank * 4);
        out << "a" << symbols.baseID(view->base);
        write_array_subscription(scope, *view, out, true, hidden_axis);
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
            if (not v.isConstant()) {
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
