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

#include <limits>
#include <iomanip>
#include <boost/filesystem/operations.hpp>
#include <jitk/codegen_util.hpp>
#include <jitk/view.hpp>
#include <jitk/instruction.hpp>

using namespace std;

namespace bohrium {
namespace jitk {


// Does 'instr' reduce over the innermost axis?
// Notice, that such a reduction computes each output element completely before moving
// to the next element.
namespace {
bool sweeping_innermost_axis(InstrPtr instr) {
    if (not bh_opcode_is_sweep(instr->opcode))
        return false;
    assert(instr->operand.size() == 3);
    return instr->sweep_axis() == instr->operand[1].ndim - 1;
}
}


void spaces(std::stringstream &out, int num) {
    for (int i = 0; i < num; ++i) {
        out << " ";
    }
}

string hash_filename(size_t compilation_hash, size_t source_hash, string extension) {
    stringstream ss;
    ss << setfill ('0') << setw(sizeof(size_t)*2) << hex << compilation_hash << "_"<< source_hash << extension;
    return ss.str();
}

boost::filesystem::path write_source2file(const std::string &src,
                                          const boost::filesystem::path &dir,
                                          const std::string &filename,
                                          bool verbose) {
    boost::filesystem::path srcfile = dir;
    srcfile /= filename;
    ofstream ofs(srcfile.string());
    ofs << src;
    ofs.flush();
    ofs.close();
    if (verbose) {
        cout << "Write source " << srcfile << endl;
    }
    return srcfile;
}

boost::filesystem::path get_tmp_path(const ConfigParser &config) {
    boost::filesystem::path tmp_path;
    const string tmp_dir = config.defaultGet<string>("tmp_dir", "");
    if (tmp_dir.empty()) {
        tmp_path = boost::filesystem::temp_directory_path();
    } else {
        tmp_path = boost::filesystem::path(tmp_dir);
    }
    return tmp_path / boost::filesystem::unique_path("bh_%%%%");
}

void create_directories(const boost::filesystem::path &path) {
    constexpr int tries = 5;
    for (int i = 1; i <= tries; ++i) {
        try {
            boost::filesystem::create_directories(path);
            return;
        } catch (boost::filesystem::filesystem_error &e) {
            if (i == tries) {
                throw;
            }
            cout << "Warning: " << e.what() << " (" << i << " attempt)" << endl;
        }
    }
}

pair<uint32_t, uint32_t> work_ranges(uint64_t work_group_size, int64_t block_size) {
    if (numeric_limits<uint32_t>::max() <= work_group_size or
        numeric_limits<uint32_t>::max() <= block_size or
        block_size < 0) {
        throw runtime_error("work_ranges(): sizes cannot fit in a uint32_t!");
    }
    const auto lsize = (uint32_t) work_group_size;
    const auto rem   = (uint32_t) block_size % lsize;
    const auto gsize = (uint32_t) block_size + (rem==0?0:(lsize-rem));
    return make_pair(gsize, lsize);
}

void write_kernel_function_arguments(const SymbolTable &symbols,
                                     std::function<const char *(bh_type type)> type_writer,
                                     stringstream &ss,
                                     const char *array_type_prefix,
                                     const bool fortran_style_param) {
    // We create the comma separated list of args and saves it in `stmp`
    stringstream stmp;
    for (size_t i=0; i < symbols.getParams().size(); ++i) {
        bh_base *b = symbols.getParams()[i];
        if (array_type_prefix != nullptr) {
            stmp << array_type_prefix << " ";
        }
        if (not fortran_style_param) { // In fortran every parameter is a pointer
            stmp << type_writer(b->type) << " * __restrict__ ";
        }
        stmp << "a" << symbols.baseID(b) << ", ";
    }
    for (const bh_view *view: symbols.offsetStrideViews()) {
        stmp << type_writer(bh_type::UINT64);
        stmp << " vo" << symbols.offsetStridesID(*view) << ", ";
        for (int i=0; i<view->ndim; ++i) {
            stmp << type_writer(bh_type::UINT64);
            stmp << " vs" << symbols.offsetStridesID(*view) << "_" << i << ", ";
        }
    }
    if (not symbols.constIDs().empty()) {
        for (auto it = symbols.constIDs().begin(); it != symbols.constIDs().end(); ++it) {
            const InstrPtr &instr = *it;
            stmp << "const " << type_writer(instr->constant.type);
            stmp << " c" << symbols.constID(*instr) << ", ";
        }
    }
    // And then we write `stmp` into `ss` excluding the last comma
    const string strtmp = stmp.str();
    if (strtmp.empty()) {
        ss << "()";
    } else {
        ss << "(";
        ss << strtmp.substr(0, strtmp.size()-2); // Excluding the last comma
        ss << ")";
    }
}


void write_loop_block(const SymbolTable &symbols,
                      const Scope *parent_scope,
                      const LoopB &block,
                      const ConfigParser &config,
                      const vector<const LoopB *> &threaded_blocks,
                      bool opencl,
                      bool fortran,
                      std::function<const char *(bh_type type)> type_writer,
                      std::function<void (const SymbolTable &symbols,
                                          Scope &scope,
                                          const LoopB &block,
                                          const ConfigParser &config,
                                          bool loop_is_peeled,
                                          const std::vector<const LoopB *> &threaded_blocks,
                                          std::stringstream &out)> head_writer,
                      std::stringstream &declares,
                      std::stringstream &out) {

    if (block.isSystemOnly()) {
        if (fortran) {
            out << "! Removed loop with only system instructions\n";
        } else {
            out << "// Removed loop with only system instructions\n";
        }
        return;
    }
    spaces(out, 4 + block.rank*4);

    // Let's find the local temporary arrays and the arrays to scalar replace
    const set<bh_base *> &local_tmps = block.getLocalTemps();

    // Let's scalar replace reduction outputs that reduces over the innermost axis
    vector<const bh_view*> scalar_replaced_reduction_outputs;
    for (const InstrPtr &instr: block._sweeps) {
        if (bh_opcode_is_reduction(instr->opcode) and sweeping_innermost_axis(instr)) {
            if (local_tmps.find(instr->operand[0].base) == local_tmps.end() and
                    (parent_scope == nullptr or parent_scope->isArray(instr->operand[0]))) {
                scalar_replaced_reduction_outputs.push_back(&instr->operand[0]);
            }
        }
    }

    // Let's scalar replace input-only arrays that are used multiple times
    vector<const bh_view*> scalar_replaced_input_only;
    {
        const vector<InstrPtr> block_instr_list = block.getAllInstr();
        // We have to ignore output arrays and arrays that are accumulated
        set<bh_base *> ignore_bases;
        for (const InstrPtr &instr: block_instr_list) {
            if (not instr->operand.empty()) {
                ignore_bases.insert(instr->operand[0].base);
            }
            if (bh_opcode_is_accumulate(instr->opcode)) {
                ignore_bases.insert(instr->operand[1].base);
            }
        }
        // First we add a valid view to the set of 'candidates' and if we encounter the view again
        // we add it to the 'scalar_replaced_input_only'
        set<bh_view> candidates;
        for (const InstrPtr &instr: block_instr_list) {
            for(size_t i=1; i < instr->operand.size(); ++i) {
                const bh_view &input = instr->operand[i];
                if ((not bh_is_constant(&input)) and ignore_bases.find(input.base) == ignore_bases.end()) {
                    if (local_tmps.find(input.base) == local_tmps.end() and
                        (parent_scope == nullptr or parent_scope->isArray(input))) {
                        if (util::exist(candidates, input)) { // 'input' is used multiple times
                            scalar_replaced_input_only.push_back(&input);
                        } else {
                            candidates.insert(input);
                        }
                    }
                }
            }
        }
    }

    // And then create the scope
    Scope scope(symbols, parent_scope, local_tmps, scalar_replaced_reduction_outputs,
                scalar_replaced_input_only, config);

    // When a reduction output is a scalar (e.g. because of array contraction or scalar replacement),
    // it should be declared before the for-loop
    for (const InstrPtr &instr: block._sweeps) {
        if (bh_opcode_is_reduction(instr->opcode)) {
            const bh_view &output = instr->operand[0];
            if (not scope.isDeclared(output) and not scope.isArray(output)) {
                // Let's write the declaration of the scalar variable
                scope.writeDeclaration(output, type_writer(output.base->type), declares);
                out << "\n";
                spaces(out, 4 + block.rank * 4);
            }
        }
    }

    // Find indexes we will declare later
    vector<const bh_view*> indexes;
    {
        set<bh_view, idx_less> candidates;
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
    }

    // We might not have to loop "peel" if all reduction have an identity value and writes to a scalar
    bool need_to_peel = false;
    {
        for (const InstrPtr &instr: block._sweeps) {
            const bh_view &v = instr->operand[0];
            if (not (has_reduce_identity(instr->opcode) and (scope.isScalarReplaced(v) or scope.isTmp(v.base)))) {
                need_to_peel = true;
                break;
            }
        }
    }

    // When not peeling, we need a neutral initial reduction value
    if (not need_to_peel) {
        for (const InstrPtr &instr: block._sweeps) {
            const bh_view &view = instr->operand[0];
            if (not scope.isArray(view) and not scope.isDeclared(view)) {
                scope.writeDeclaration(view, type_writer(view.base->type), declares);
                out << "\n";
                spaces(out, 4 + block.rank * 4);
            }
            scope.getName(view, out);
            out << " = ";
            write_reduce_identity(instr->opcode, view.base->type, out);
            out << ";\n";
            spaces(out, 4 + block.rank * 4);
        }
    }

    // If this block is sweeped, we will "peel" the for-loop such that the
    // sweep instruction is replaced with BH_IDENTITY in the first iteration
    if (block._sweeps.size() > 0 and need_to_peel) {
        Scope peeled_scope(scope);
        LoopB peeled_block(block);
        for (const InstrPtr instr: block._sweeps) {
            // The input is the same as in the sweep
            bh_instruction sweep_instr(BH_IDENTITY, {instr->operand[0], instr->operand[1]});

            // But the output needs an extra dimension when we are reducing to a non-scalar
            if (bh_opcode_is_reduction(instr->opcode) and instr->operand[1].ndim > 1) {
                sweep_instr.operand[0].insert_axis(instr->constant.get_int64(), 1, 0);
            }
            peeled_block.replaceInstr(instr, sweep_instr);
        }
        string itername;
        {stringstream t; t << "i" << block.rank; itername = t.str();}
        if (fortran) {
            out << "! Peeled loop, 1. sweep iteration\n";
            spaces(out, 8 + block.rank*4);
            out << itername << " = 0;\n";
        } else {
            out << "{ // Peeled loop, 1. sweep iteration\n";
            spaces(out, 8 + block.rank*4);
            declares << type_writer(bh_type::UINT64) << " " << itername << " = 0;\n";
        }

        // Write temporary and scalar replaced array declarations
        for (const InstrPtr &instr: block.getLocalInstr()) {
            for (const bh_view *view: instr->get_views()) {
                if (not peeled_scope.isDeclared(*view)) {
                    if (peeled_scope.isTmp(view->base)) {
                        spaces(out, 8 + block.rank * 4);
                        if (not fortran) {
                            peeled_scope.writeDeclaration(*view, type_writer(view->base->type), declares);
                            out << "\n";
                        }
                    } else if (peeled_scope.isScalarReplaced_R(*view)) {
                        spaces(out, 8 + block.rank * 4);
                        if (not fortran) {
                            peeled_scope.writeDeclaration(*view, type_writer(view->base->type), declares);
                        }
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
                spaces(out, 8 + block.rank * 4);
                if (not fortran) {
                    peeled_scope.writeIdxDeclaration(*view, type_writer(bh_type::UINT64), declares);
                }
                out << "\n";
            }
        }
        out << "\n";
        for (const Block &b: peeled_block._block_list) {
            if (b.isInstr()) {
                if (b.getInstr() != nullptr and not bh_opcode_is_system(b.getInstr()->opcode)) {
                    spaces(out, 4 + b.rank()*4);
                    write_instr(peeled_scope, *b.getInstr(), out, opencl);
                }
            } else {
                write_loop_block(symbols, &peeled_scope, b.getLoop(), config, threaded_blocks, opencl, fortran,
                                 type_writer, head_writer, declares, out);
            }
        }
        spaces(out, 4 + block.rank*4);
        if (not fortran) {
            out << "}";
        }
        out << "\n";
        spaces(out, 4 + block.rank*4);
    }

    // Write the for-loop header
    head_writer(symbols, scope, block, config, need_to_peel, threaded_blocks, out);

    // Write temporary and scalar replaced array declarations
    for (const InstrPtr &instr: block.getLocalInstr()) {
        for (const bh_view *view: instr->get_views()) {
            if (not scope.isDeclared(*view)) {
                if (scope.isTmp(view->base)) {
                    spaces(out, 8 + block.rank * 4);
                    scope.writeDeclaration(*view, type_writer(view->base->type), declares);
                    out << "\n";
                } else if (scope.isScalarReplaced_R(*view)) {
                    spaces(out, 8 + block.rank * 4);
                    scope.writeDeclaration(*view, type_writer(view->base->type), declares);
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
            spaces(out, 8 + block.rank * 4);
            scope.writeIdxDeclaration(*view, type_writer(bh_type::UINT64), declares);
            out << "\n";
        }
    }

    // Write the for-loop body
    // The body in OpenCL and OpenMP are very similar but OpenMP might need to insert "#pragma omp atomic/critical"
    if (opencl) {
        for (const Block &b: block._block_list) {
            if (b.isInstr()) { // Finally, let's write the instruction
                if (b.getInstr() != NULL and not bh_opcode_is_system(b.getInstr()->opcode)) {
                    spaces(out, 4 + b.rank()*4);
                    write_instr(scope, *b.getInstr(), out, true);
                }
            } else {
                write_loop_block(symbols, &scope, b.getLoop(), config, threaded_blocks, opencl, fortran, type_writer,
                                 head_writer, declares, out);
            }
        }
    } else {
        for (const Block &b: block._block_list) {
            if (b.isInstr()) { // Finally, let's write the instruction
                const InstrPtr instr = b.getInstr();
                if (not bh_opcode_is_system(instr->opcode)) {
                    if (instr->operand.size() > 0) {
                        if (scope.isOpenmpAtomic(instr->operand[0])) {
                            spaces(out, 4 + b.rank()*4);
                            out << "#pragma omp atomic\n";
                        } else if (scope.isOpenmpCritical(instr->operand[0])) {
                            spaces(out, 4 + b.rank()*4);
                            out << "#pragma omp critical\n";
                        }
                    }
                    spaces(out, 4 + b.rank()*4);
                    write_instr(scope, *instr, out);
                }
            } else {
                write_loop_block(symbols, &scope, b.getLoop(), config, threaded_blocks, opencl, fortran, type_writer,
                                 head_writer, declares, out);
            }
        }
    }
    spaces(out, 4 + block.rank*4);
    if (not fortran) {
        out << "}";
    }
    out << "\n";

    // Let's copy the scalar replaced reduction outputs back to the original array
    for (const bh_view *view: scalar_replaced_reduction_outputs) {
        spaces(out, 4 + block.rank*4);
        out << "a" << symbols.baseID(view->base);
        write_array_subscription(scope, *view, out, true);
        out << " = ";
        scope.getName(*view, out);
        out << ";\n";
    }
}

// Handle the extension methods within the 'bhir'
void util_handle_extmethod(component::ComponentImpl *self,
                           bh_ir *bhir,
                           std::map<bh_opcode, extmethod::ExtmethodFace> &extmethods) {

    std::vector<bh_instruction> instr_list;
    for (bh_instruction &instr: bhir->instr_list) {
        auto ext = extmethods.find(instr.opcode);
        if (ext != extmethods.end()) {
            bh_ir b;
            b.instr_list = instr_list;
            self->execute(&b); // Execute the instructions up until now
            instr_list.clear();
            ext->second.execute(&instr, NULL); // Execute the extension method
        } else {
            instr_list.push_back(instr);
        }
    }
    bhir->instr_list = instr_list;
}

} // jitk
} // bohrium
