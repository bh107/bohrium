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
#include <stdexcept>
#include <map>

#include <inttypes.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>

#define BH_TIMING_SUM
#include "bh_timing.hpp"

#include "timevault.hpp"
#include "utils.hpp"
#include "engine.hpp"
#include "kp_rt.h"

const char TAG[] = "Component";

#include <bh_component.hpp>
#include <bh_extmethod.hpp>

using namespace std;
using namespace bohrium;
using namespace component;
using namespace extmethod;

typedef vector<bh_instruction> instr_iter;
typedef vector<bh_ir_kernel>::iterator krnl_iter;

class Impl : public ComponentImpl {
  private:
    // This is where the actual engine implementation is
    kp::engine::Engine* engine = NULL;

    // Timing ID for timing of execute()
    bh_intp exec_timing;
    bool timing;
    size_t exec_count = 0;
    map<bh_opcode, ExtmethodFace> extensions;
  public:
    Impl(unsigned int stack_level);
    ~Impl() {
        delete engine;
        if (timing) {
            bh_timer_finalize(exec_timing);
        }
    }
    void execute(bh_ir *bhir);
    void extmethod(const string &name, bh_opcode opcode) {
        // ExtmethodFace does not have a default or copy constructor thus
        // we have to use its move constructor.
        extensions.insert(make_pair(opcode, ExtmethodFace(config, name)));
    }
};

extern "C" ComponentImpl* create(unsigned int stack_level) {
    return new Impl(stack_level);
}
extern "C" void destroy(ComponentImpl* self) {
    delete self;
}

Impl::Impl(unsigned int stack_level) : ComponentImpl(stack_level) {
    char* env = getenv("BH_FUSE_MODEL");                    // Set the fuse-model
    if (NULL != env) {
        string env_str(env);
        if (!env_str.compare("same_shape_generate_1dreduce")) {
            fprintf(stderr, "[CPU-VE] Warning! unsupported fuse model: '%s"
                    "', it may not work.\n", env);
        }
    } else {
        setenv("BH_FUSE_MODEL", "SAME_SHAPE_GENERATE_1DREDUCE", 1);
    }

    //
    //  Get engine parameters
    //
    timing = config.defaultGet<bool>("timing", false);
    bh_intp bind = config.get<bh_intp>("bind");
    bh_intp vcache_size = config.get<bh_intp>("vcache_size");
    bool preload = config.defaultGet<bool>("preload", false);

    bh_intp jit_level   = config.get<bh_intp>("jit_level");
    bool jit_dumpsrc    = config.defaultGet<bool>("jit_dumpsrc", false);
    bh_intp jit_offload = config.get<bh_intp>("jit_offload");

    string compiler_cmd = config.get<string>("compiler_cmd");
    string compiler_inc = config.get<string>("compiler_inc");
    string compiler_lib = config.get<string>("compiler_lib");
    string compiler_flg = config.get<string>("compiler_flg");
    string compiler_ext = config.get<string>("compiler_ext");

    string object_path   = config.get<string>("object_path");
    string template_path = config.get<string>("template_path");
    string kernel_path   = config.get<string>("kernel_path");

    // Initialize execute(...) timer
    if (timing) {
        exec_timing = bh_timer_new("[VE-CPU] Execution");
    }

    //
    //  Set JIT-parameters based on JIT-LEVEL
    //
    bool jit_enabled     = false;
    bool jit_fusion      = false;
    bool jit_contraction = false;

    switch(jit_level) {
        case 0:                     // Disable JIT, rely on preload.
            preload         = true;
            jit_enabled     = false;
            jit_dumpsrc     = false;
            jit_fusion      = false;
            jit_contraction = false;
            break;

        case 3:                     // SIJ + Fusion + Contraction
            jit_contraction = true;
        case 2:                     // SIJ + Fusion
            jit_fusion = true;
        case 1:                     // SIJ
        default:
            jit_enabled = true;
            break;
    }

    // Make sure that kernel and object path exists
    // TODO: This is anti-portable and should be fixed.
    mkdir(kernel_path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    mkdir(object_path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

    // Construct architecture id for object-store
    char* host_cstr = new char[200];
    kp_set_host_text(host_cstr);
    string host_str(host_cstr);
    delete[] host_cstr;

    string arch_id = kp::core::hash_text(host_str);
    string object_directory;            // Subfolder of object_path
    string sep("/");                    // TODO: Portable file-separator

    object_directory = object_path + sep + arch_id;

    // Create object-directory for target/arch_id
    if (access(object_directory.c_str(), F_OK)) {
        int err = mkdir(object_directory.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        if (err) {
            if (access(object_directory.c_str(), F_OK)) {
                stringstream ss;
                ss << "CPU-VE: create_directory(" << object_directory << "), ";
                ss << "failed. And it does not seem to exist. This directory ";
                ss << "is needed for JIT-compilation.";
                throw runtime_error(ss.str());
            }
        }
    }

    //
    // VROOM VROOM VROOOOOOMMMM!!! VROOOOM!!
    engine = new kp::engine::Engine(
        (kp_thread_binding)bind,
        (size_t)vcache_size,
        preload,
        jit_enabled,
        jit_dumpsrc,
        jit_fusion,
        jit_contraction,
        (size_t)jit_offload,
        string(compiler_cmd),
        string(compiler_inc),
        string(compiler_lib),
        string(compiler_flg),
        string(compiler_ext),
        string(object_directory),
        string(template_path),
        string(kernel_path)
    );

    if (getenv("BH_VE_CPU_INFO")) { // Print out engine configuration
        cout << engine->text() << endl;
    }
}

void Impl::execute(bh_ir* bhir) {
    bh_uint64 timestamp = 0;
    if (timing) {
        timestamp = bh_timer_stamp();
    }
    bh_error res = BH_SUCCESS;

    exec_count++;
    DEBUG(TAG, "EXEC #" << exec_count);

    //
    // Map Bohrium program representation to CAPE
    uint64_t program_size = bhir->instr_list.size();
    kp::core::Program tac_program(program_size);                  // Program
    kp::core::SymbolTable symbol_table(program_size*6+2);         // SymbolTable

    kp::core::instrs_to_tacs(*bhir, tac_program, symbol_table);   // Map instructions to tac and symbol_table.

    kp::core::Block block(symbol_table, tac_program);             // Construct a block

    //
    //  Map bh_kernels to Blocks one at a time and execute them.
    for(krnl_iter krnl = bhir->kernel_list.begin();
        krnl != bhir->kernel_list.end();
        ++krnl) {

        block.clear();                                          // Reset the block
        block.compose(*krnl, (bool)engine->jit_contraction());  // Compose it based on kernel

        TIMER_DETAILED
        if ((block.omask() & KP_EXTENSION)>0) {         // Extension-Instruction-Execute (EIE)
            TIMER_START
            kp_tac& tac = block.tac(0);
            auto ext = extensions.find(static_cast<bh_instruction*>(tac.ext)->opcode);
            if (ext != extensions.end()) {
                ext->second.execute(static_cast<bh_instruction*>(tac.ext), NULL);
            }
            TIMER_STOP(block.text_compact());
        } else if ((engine->jit_fusion()) ||
                   (block.narray_tacs() == 0)) {        // Multi-Instruction-Execute (MIE)
            DEBUG(TAG, "Multi-Instruction-Execute BEGIN");

            TIMER_START
            res = engine->process_block(block);
            TIMER_STOP(block.text_compact());

            if (BH_SUCCESS != res) {
                throw runtime_error("VE-CPU: fatal error");
            }
            DEBUG(TAG, "Muilti-Instruction-Execute END");
        } else {                                        // Single-Instruction-Execute (SIE)
            DEBUG(TAG, "Single-Instruction-Execute BEGIN");
            for(std::vector<uint64_t>::const_iterator idx_it = krnl->instr_indexes().begin();
                idx_it != krnl->instr_indexes().end();
                ++idx_it) {

                block.clear();                          // Reset the block
                block.compose(*krnl, (size_t)*idx_it);  // Compose based on a single instruction

                TIMER_START
                res = engine->process_block(block);
                TIMER_STOP(block.text_compact());

                if (BH_SUCCESS != res) {
                    throw runtime_error("VE-CPU: fatal error");
                }
            }
            DEBUG(TAG, "Single-Instruction-Execute END");
        }
    }

    if (timing) {
        bh_timer_add(exec_timing, timestamp, bh_timer_stamp());
    }
}


