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
#include "bh_ve_cpu.h"

#include "timevault.hpp"
#include "utils.hpp"
#include "engine.hpp"
#include "kp_rt.h"

using namespace std;
const char TAG[] = "Component";

static bh_component myself;

//
// This is where the actual engine implementation is
static kp::engine::Engine* engine = NULL;

// Timing ID for timing of execute()
static bh_intp exec_timing;
static bool timing;
static size_t exec_count = 0;
static map<bh_opcode, bh_extmethod_impl> extensions;

typedef vector<bh_instruction> instr_iter;
typedef vector<bh_ir_kernel>::iterator krnl_iter;

/* Component interface: init (see bh_component.h) */
bh_error bh_ve_cpu_init(const char *name)
{
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
    if (BH_SUCCESS != bh_component_init(&myself, name)) {   // Initialize engine
        fprintf(stderr, "[CPU-VE] Failed initializing component\n");
        return BH_ERROR;
    }
    if (0 != myself.nchildren) {                            // Check stack
        fprintf(stderr, "[CPU-VE] Unexpected number of children, must be 0\n");
        return BH_ERROR;
    }

    //
    //  Get engine parameters
    //
    bh_intp bind;
    bh_intp vcache_size;
    bool preload;

    bh_intp jit_level;
    bool jit_dumpsrc;
    bh_intp jit_offload;

    char* compiler_cmd = NULL;
    char* compiler_inc = NULL;
    char* compiler_lib = NULL;
    char* compiler_flg = NULL;
    char* compiler_ext = NULL;

    char* object_path = NULL;
    char* template_path = NULL;
    char* kernel_path = NULL;

    if ((BH_SUCCESS!=bh_component_config_bool_option(&myself, "timing", &timing))                   or \
        (BH_SUCCESS!=bh_component_config_int_option(&myself, "bind", 0, 2, &bind))                  or \
        (BH_SUCCESS!=bh_component_config_int_option(&myself, "vcache_size", 0, 100, &vcache_size))  or \
        (BH_SUCCESS!=bh_component_config_bool_option(&myself, "preload", &preload))                 or \
        (BH_SUCCESS!=bh_component_config_int_option(&myself, "jit_level", 0, 3, &jit_level))        or \
        (BH_SUCCESS!=bh_component_config_bool_option(&myself, "jit_dumpsrc", &jit_dumpsrc))         or \
        (BH_SUCCESS!=bh_component_config_int_option(&myself, "jit_offload", 0, 2, &jit_offload))    or \
        (BH_SUCCESS!=bh_component_config_string_option(&myself, "compiler_cmd", &compiler_cmd))     or \
        (BH_SUCCESS!=bh_component_config_string_option(&myself, "compiler_inc", &compiler_inc))     or \
        (BH_SUCCESS!=bh_component_config_string_option(&myself, "compiler_lib", &compiler_lib))     or \
        (BH_SUCCESS!=bh_component_config_string_option(&myself, "compiler_flg", &compiler_flg))     or \
        (BH_SUCCESS!=bh_component_config_string_option(&myself, "compiler_ext", &compiler_ext))     or \
        (BH_SUCCESS!=bh_component_config_path_option(&myself, "object_path", &object_path))         or \
        (BH_SUCCESS!=bh_component_config_path_option(&myself, "kernel_path", &kernel_path))         or \
        (BH_SUCCESS!=bh_component_config_path_option(&myself, "template_path", &template_path))) {
        return BH_ERROR;
    }

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

	//
    // Make sure that kernel and object path exists
	// TODO: This is anti-portable and should be fixed.
    mkdir(kernel_path, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    mkdir(object_path, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

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

    return BH_SUCCESS;
}

/* Component interface: execute (see bh_component.h) */
bh_error bh_ve_cpu_execute(bh_ir* bhir)
{
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
            map<bh_opcode, bh_extmethod_impl>::iterator ext;
            ext = extensions.find(static_cast<bh_instruction*>(tac.ext)->opcode);
            if (ext != extensions.end()) {
                bh_extmethod_impl extmethod = ext->second;
                res = extmethod(static_cast<bh_instruction*>(tac.ext), NULL);
                if (BH_SUCCESS != res) {
                    fprintf(stderr, "Unhandled error returned by extmethod(...) \n");
                    return res;
                }
            }
            TIMER_STOP(block.text_compact());
        } else if ((engine->jit_fusion()) ||
                   (block.narray_tacs() == 0)) {        // Multi-Instruction-Execute (MIE)
            DEBUG(TAG, "Multi-Instruction-Execute BEGIN");

            TIMER_START
            res = engine->process_block(block);
            TIMER_STOP(block.text_compact());

            if (BH_SUCCESS != res) {
                return res;
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
                    return res;
                }
            }
            DEBUG(TAG, "Single-Instruction-Execute END");
        }
    }

    if (timing) {
        bh_timer_add(exec_timing, timestamp, bh_timer_stamp());
    }

    return res;
}

/* Component interface: shutdown (see bh_component.h) */
bh_error bh_ve_cpu_shutdown(void)
{
    bh_component_destroy(&myself);

    delete engine;
    engine = NULL;

    if (timing) {
        bh_timer_finalize(exec_timing);
    }

    return BH_SUCCESS;
}

/* Component interface: extmethod (see bh_component.h) */
bh_error bh_ve_cpu_extmethod(const char *name, bh_opcode opcode)
{
    bh_extmethod_impl extmethod;
    bh_error err = bh_component_extmethod(&myself, name, &extmethod);
    if (err != BH_SUCCESS) {
        return err;
    }

    if (extensions.find(opcode) != extensions.end()) {
        fprintf(stderr, "[CPU-VE] Warning, multiple registrations of the same"
               "extension method '%s' (opcode: %d)\n", name, (int)opcode);
    }
    extensions[opcode] = extmethod;

    return BH_SUCCESS;
}
