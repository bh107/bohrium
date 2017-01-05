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

#include <jitk/codegen_util.hpp>
#include <jitk/instruction.hpp>

using namespace std;

namespace bohrium {
namespace jitk {


// Does 'instr' reduce over the innermost axis?
// Notice, that such a reduction computes each output element completely before moving
// to the next element.
bool sweeping_innermost_axis(InstrPtr instr) {
    if (not bh_opcode_is_sweep(instr->opcode))
        return false;
    assert(bh_noperands(instr->opcode) == 3);
    return instr->sweep_axis() == instr->operand[1].ndim-1;
}

void write_loop_block(BaseDB &base_ids,
                      const LoopB &block,
                      const ConfigParser &config,
                      const vector<const LoopB *> &threaded_blocks,
                      bool opencl,
                      std::function<const char *(bh_type type)> type_writer,
                      std::function<void (BaseDB &base_ids,
                                          const LoopB &block,
                                          const ConfigParser &config,
                                          bool loop_is_peeled,
                                          const std::vector<const LoopB *> &threaded_blocks,
                                          std::stringstream &out)> head_writer,
                      std::stringstream &out) {
    spaces(out, 4 + block.rank*4);

    if (block.isSystemOnly()) {
        out << "// Removed loop with only system instructions" << endl;
        return;
    }

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
            base_ids.writeDeclaration(base, type_writer(base->type), out);
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
                base_ids.writeDeclaration(base, type_writer(base->type), out);
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

    // Get local temporary arrays as a vector sorted by the ID
    vector<bh_base*> local_tmps;
    {
        const set<bh_base *> t = block.getLocalTemps();
        for (bh_base *base: base_ids.getBases()) {
            if (util::exist(t, base)) {
                local_tmps.push_back(base);
            }
        }
    }

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
        out << type_writer(BH_UINT64) << " " << itername << " = 0;\n";
        // Write temporary array declarations
        for (bh_base* base: local_tmps) {
            assert(base_ids_tmp.isTmp(base));
            if (not base_ids_tmp.isLocallyDeclared(base)) {
                spaces(out, 8 + block.rank * 4);
                base_ids_tmp.writeDeclaration(base, type_writer(base->type), out);
                out << "\n";
            }
        }
        out << "\n";
        for (const Block &b: peeled_block._block_list) {
            if (b.isInstr()) {
                if (b.getInstr() != NULL) {
                    spaces(out, 4 + b.rank()*4);
                    write_instr(base_ids_tmp, *b.getInstr(), out, opencl);
                }
            } else {
                write_loop_block(base_ids_tmp, b.getLoop(), config, threaded_blocks, opencl, type_writer, head_writer, out);
            }
        }
        spaces(out, 4 + block.rank*4);
        out << "}\n";
        spaces(out, 4 + block.rank*4);
    }

    // Write the for-loop header
    head_writer(base_ids, block, config, need_to_peel, threaded_blocks, out);

    // Write temporary array declarations
    for (bh_base* base: local_tmps) {
        assert(base_ids.isTmp(base));
        if (not base_ids.isLocallyDeclared(base)) {
            spaces(out, 8 + block.rank * 4);
            base_ids.writeDeclaration(base, type_writer(base->type), out);
            out << "\n";
        }
    }

    // Write the for-loop body
    // The body in OpenCL and OpenMP are very similar but OpenMP might need to insert "#pragma omp atomic/critical"
    if (opencl) {
        for (const Block &b: block._block_list) {
            if (b.isInstr()) { // Finally, let's write the instruction
                if (b.getInstr() != NULL) {
                    spaces(out, 4 + b.rank()*4);
                    write_instr(base_ids, *b.getInstr(), out, true);
                }
            } else {
                write_loop_block(base_ids, b.getLoop(), config, threaded_blocks, opencl, type_writer, head_writer, out);
            }
        }
    } else {
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
                write_loop_block(base_ids, b.getLoop(), config, threaded_blocks, opencl, type_writer, head_writer, out);
            }
        }
    }
    spaces(out, 4 + block.rank*4);
    out << "}\n";

    // Let's copy the scalar replacement back to the original array
    for (const bh_view &view: scalar_replacements) {
        spaces(out, 4 + block.rank*4);
        const size_t id = base_ids[view.base];
        out << "a" << id;
        write_array_subscription(view, out);
        out << " = s" << id << ";\n";
        base_ids.eraseScalarReplacement(view.base); // It is not scalar replaced anymore
    }
}


} // jitk
} // bohrium
