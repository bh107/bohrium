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
#include <jitk/statistics.hpp>
#include <jitk/dtype.hpp>
#include <jitk/apply_fusion.hpp>

#include "engine_openmp.hpp"
#include "openmp_util.hpp"

using namespace bohrium;
using namespace jitk;
using namespace component;
using namespace std;

namespace {
class Impl : public ComponentImpl {
  private:
    // Some statistics
    Statistics stat;
    // Fuse cache
    FuseCache fcache;
    // Teh OpenMP engine
    EngineOpenMP engine;
    // Known extension methods
    map<bh_opcode, extmethod::ExtmethodFace> extmethods;
    //Allocated base arrays
    set<bh_base*> _allocated_bases;

  public:
    Impl(int stack_level) : ComponentImpl(stack_level),
                            stat(config.defaultGet("prof", false)),
                            fcache(stat), engine(config, stat) {}
    ~Impl();
    void execute(bh_ir *bhir);
    void extmethod(const string &name, bh_opcode opcode) {
        // ExtmethodFace does not have a default or copy constructor thus
        // we have to use its move constructor.
        extmethods.insert(make_pair(opcode, extmethod::ExtmethodFace(config, name)));
    }

    // Implement the handle of extension methods
    void handle_extmethod(bh_ir *bhir) {
        util_handle_extmethod(this, bhir, extmethods);
    }

    // The following methods implements the methods required by jitk::handle_execution()

    // Write the OpenMP kernel
    void write_kernel(Kernel &kernel, const SymbolTable &symbols, const ConfigParser &config,
                      const vector<const LoopB *> &threaded_blocks,
                      const vector<const bh_view*> &offset_strides, stringstream &ss);

    // Returns the blocks that can be parallelized in 'kernel' (incl. sub-blocks)
    vector<const LoopB*> find_threaded_blocks(Kernel &kernel) {
        vector<const LoopB*> threaded_blocks = {&kernel.block};
        return threaded_blocks;
    }

    // Handle messages from parent
    string message(const string &msg) {
        if (msg == "statistic_enable_and_reset") {
            stat = Statistics(true, config.defaultGet("prof", false));
        } else if (msg == "statistic") {
            stringstream ss;
            stat.write("OpenMP", "", ss);
            return ss.str();
        }
        return "";
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
    if (stat.print_on_exit) {
        stat.write("OpenMP", config.defaultGet<std::string>("prof_filename", ""), cout);
    }
}

// Writing the OpenMP header, which include "parallel for" and "simd"
void write_openmp_header(const SymbolTable &symbols, Scope &scope, const LoopB &block, const ConfigParser &config, stringstream &out) {
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
            assert(instr->operand.size() == 3);
            const bh_view &view = instr->operand[0];
            if (openmp_reduce_compatible(instr->opcode) and (scope.isScalarReplaced(view) or scope.isTmp(view.base))) {
                openmp_reductions.push_back(instr);
            } else if (openmp_atomic_compatible(instr->opcode)) {
                scope.insertOpenmpAtomic(view);
            } else {
                scope.insertOpenmpCritical(view);
            }
        }
    }

    // "OpenMP SIMD" goes to the innermost loop (which might also be the outermost loop)
    if (enable_simd and block.isInnermost() and simd_compatible(block, scope)) {
        ss << " simd";
        if (block.rank > 0) { //NB: avoid multiple reduction declarations
            for (const InstrPtr instr: block._sweeps) {
                openmp_reductions.push_back(instr);
            }
        }
    }

    //Let's write the OpenMP reductions
    for (const InstrPtr instr: openmp_reductions) {
        assert(instr->operand.size() == 3);
        ss << " reduction(" << openmp_reduce_symbol(instr->opcode) << ":";
        scope.getName(instr->operand[0], ss);
        ss << ")";
    }
    const string ss_str = ss.str();
    if(not ss_str.empty()) {
        out << "#pragma omp" << ss_str << "\n";
        spaces(out, 4 + block.rank*4);
    }
}

// Writes the OpenMP specific for-loop header
void loop_head_writer(const SymbolTable &symbols, Scope &scope, const LoopB &block, const ConfigParser &config, bool loop_is_peeled,
                      const vector<const LoopB *> &threaded_blocks, stringstream &out) {

    // Let's write the OpenMP loop header
    {
        int64_t for_loop_size = block.size;
        if (block._sweeps.size() > 0 and loop_is_peeled) // If the for-loop has been peeled, its size is one less
            --for_loop_size;
        // No need to parallel one-sized loops
        if (for_loop_size > 1) {
            write_openmp_header(symbols, scope, block, config, out);
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

void Impl::write_kernel(Kernel &kernel, const SymbolTable &symbols, const ConfigParser &config,
                        const vector<const LoopB *> &threaded_blocks,
                        const vector<const bh_view*> &offset_strides, stringstream &ss) {

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
    write_c99_dtype_union(ss); // We always need to declare the union of all constant data types
    ss << "\n";

    // Write the header of the execute function
    ss << "void execute";
    write_kernel_function_arguments(kernel, symbols, offset_strides, write_c99_type, ss, NULL, false);

    // Write the block that makes up the body of 'execute()'
    ss << "{\n";
    write_loop_block(symbols, NULL, kernel.block, config, {}, false, write_c99_type, loop_head_writer, ss);
    ss << "}\n\n";

    // Write the launcher function, which will convert the data_list of void pointers
    // to typed arrays and call the execute function
    {
        ss << "void launcher(void* data_list[], uint64_t offset_strides[], union dtype constants[]) {\n";
        for(size_t i=0; i < kernel.getNonTemps().size(); ++i) {
            spaces(ss, 4);
            bh_base *b = kernel.getNonTemps()[i];
            ss << write_c99_type(b->type) << " *a" << symbols.baseID(b);
            ss << " = data_list[" << i << "];\n";
        }
        spaces(ss, 4);
        ss << "execute(";
        for(size_t i=0; i < kernel.getNonTemps().size(); ++i) {
            bh_base *b = kernel.getNonTemps()[i];
            ss << "a" << symbols.baseID(b);
            if (i+1 < kernel.getNonTemps().size()) {
                ss << ", ";
            }
        }
        uint64_t count=0;
        for (const bh_view *view: offset_strides) {
            ss << ", offset_strides[" << count++ << "]";
            for (int i=0; i<view->ndim; ++i) {
                ss << ", offset_strides[" << count++ << "]";
            }
        }
        if (symbols.constIDs().size() > 0) {
            if (kernel.getNonTemps().size() > 0) {
                ss << ", "; // If any args were written before us, we need a comma
            }
            uint64_t i=0;
            for (auto it = symbols.constIDs().begin(); it != symbols.constIDs().end();) {
                const InstrPtr &instr = *it;
                ss << "constants[" << i++ << "]." << bh_type_text(instr->constant.type);
                if (++it != symbols.constIDs().end()) { // Not the last iteration
                    ss << ", ";
                }
            }
        }
        ss << ");\n";
        ss << "}\n";
    }
}

void Impl::execute(bh_ir *bhir) {
    // Let's handle extension methods
    util_handle_extmethod(this, bhir, extmethods);

    // And then the regular instructions
    handle_execution(*this, bhir, engine, config, stat, fcache, NULL);
}
