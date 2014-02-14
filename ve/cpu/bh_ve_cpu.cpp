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
#include <string>
#include <sstream>
#include <vector>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <errno.h>
#include <unistd.h>
#include <inttypes.h>
#include <ctemplate/template.h>
#include <bh.h>
#include <bh_vcache.h>
#include "bh_ve_cpu.h"

// Execution Profile
#ifdef PROFILE
static bh_uint64 times[BH_NO_OPCODES+2]; // opcodes and: +1=malloc, +2=kernel
static bh_uint64 calls[BH_NO_OPCODES+2];
#endif

using namespace std;

static bh_component myself;
static map<bh_opcode, bh_extmethod_impl> extmethod_op2impl;

static bh_intp vcache_size  = 10;
static bh_intp jit_enabled  = 1;
static bh_intp jit_preload  = 1;
static bh_intp jit_fusion   = 0;
static bh_intp jit_optimize = 1;
static bh_intp jit_dumpsrc  = 0;

static char* compiler_cmd;   // cpu Arguments
static char* kernel_path;
static char* object_path;
static char* template_path;

typedef struct block {
    int tsig[10];               // Typesignature of the instructions
    int lmask[10];              // Layoutmask of the instructions
    bh_instruction* instr[10];  // Pointers to instructions

    tac_t* program;             // Ordered list of TACs
    block_arg_t* scope;         // Array of block arguments

    uint32_t nargs;             // Number of arguments to the block
    int length;                 // Number of tacs in program
    uint32_t omask;             // Mask of the OPERATIONS in the block
    string symbol;              // Textual representation of the block
} block_t;                      // Meta-data to construct and execute a block-function

#include "utils.cpp"
#include "block.c"
#include "operator_cexpr.c"
#include "compiler.cpp"
#include "specializer.cpp"

Compiler* target;

/* Component interface: init (see bh_component.h) */
bh_error bh_ve_cpu_init(const char *name)
{
    char *env;
    bh_error err;

    if((err = bh_component_init(&myself, name)) != BH_SUCCESS)
        return err;
    if(myself.nchildren != 0)
    {
        std::cerr << "[CPU-VE] Unexpected number of children, must be 0" << std::endl;
        return BH_ERROR;
    }

    env = getenv("BH_CORE_VCACHE_SIZE");      // Override block_size from environment-variable.
    if (NULL != env) {
        vcache_size = atoi(env);
    }
    if (0 > vcache_size) {                          // Verify it
        fprintf(stderr, "BH_CORE_VCACHE_SIZE (%ld) should be greater than zero!\n", (long int)vcache_size);
        return BH_ERROR;
    }

    env = getenv("BH_VE_CPU_JIT_ENABLED");
    if (NULL != env) {
        jit_enabled = atoi(env);
    }
    if (!((0==jit_enabled) || (1==jit_enabled))) {
        fprintf(stderr, "BH_VE_CPU_JIT_ENABLED (%ld) should 0 or 1.\n", (long int)jit_enabled);
        return BH_ERROR;
    }

    env = getenv("BH_VE_CPU_JIT_PRELOAD");
    if (NULL != env) {
        jit_preload = atoi(env);
    }
    if (!((0==jit_preload) || (1==jit_preload))) {
        fprintf(stderr, "BH_VE_CPU_JIT_PRELOAD (%ld) should 0 or 1.\n", (long int)jit_preload);
        return BH_ERROR;
    }

    env = getenv("BH_VE_CPU_JIT_FUSION");
    if (NULL != env) {
        jit_fusion = atoi(env);
    }
    if (!((0==jit_fusion) || (1==jit_fusion))) {
        fprintf(stderr, "BH_VE_CPU_JIT_FUSION (%ld) should 0 or 1.\n", (long int)jit_fusion);
        return BH_ERROR;
    }

    env = getenv("BH_VE_CPU_JIT_OPTIMIZE");
    if (NULL != env) {
        jit_optimize = atoi(env);
    }
    if (!((0==jit_optimize) || (1==jit_optimize))) {
        fprintf(stderr, "BH_VE_CPU_JIT_OPTIMIZE (%ld) should 0 or 1.\n", (long int)jit_optimize);
        return BH_ERROR;
    }

    env = getenv("BH_VE_CPU_JIT_DUMPSRC");
    if (NULL != env) {
        jit_dumpsrc = atoi(env);
    }
    if (!((0==jit_dumpsrc) || (1==jit_dumpsrc))) {
         fprintf(stderr, "BH_VE_CPU_JIT_DUMPSRC (%ld) should 0 or 1.\n", (long int)jit_dumpsrc);
        return BH_ERROR;
    }

    // Victim cache
    bh_vcache_init(vcache_size);

    // Configuration
    bh_path_option(     kernel_path,    "BH_VE_CPU_KERNEL_PATH",   "kernel_path");
    bh_path_option(     object_path,    "BH_VE_CPU_OBJECT_PATH",   "object_path");
    bh_path_option(     template_path,  "BH_VE_CPU_TEMPLATE_PATH", "template_path");
    bh_string_option(   compiler_cmd,   "BH_VE_CPU_COMPILER",      "compiler_cmd");

    if (!jit_enabled) {
        jit_preload     = 1;
        jit_fusion      = 0;
        jit_optimize    = 0;
        jit_dumpsrc     = 0;
    }

    if (false) {
        std::cout << "ENVIRONMENT {" << std::endl;
        std::cout << "  BH_CORE_VCACHE_SIZE="     << vcache_size  << std::endl;
        std::cout << "  BH_VE_CPU_JIT_ENABLED="   << jit_enabled  << std::endl;
        std::cout << "  BH_VE_CPU_JIT_PRELOAD="   << jit_preload  << std::endl;
        std::cout << "  BH_VE_CPU_JIT_FUSION="    << jit_fusion   << std::endl;
        std::cout << "  BH_VE_CPU_JIT_OPTIMIZE="  << jit_optimize << std::endl;
        std::cout << "  BH_VE_CPU_JIT_DUMPSRC="   << jit_dumpsrc  << std::endl;
        std::cout << "}" << std::endl;
    }

    // JIT machinery
    target = new Compiler(compiler_cmd, object_path, kernel_path, jit_preload);
    specializer_init();     // Code templates and opcode-specialization.

    #ifdef PROFILE
    memset(&times, 0, sizeof(bh_uint64)*(BH_NO_OPCODES+2));
    memset(&calls, 0, sizeof(bh_uint64)*(BH_NO_OPCODES+2));
    #endif

    return BH_SUCCESS;
}

/* Component interface: execute (see bh_component.h) */
bh_error bh_ve_cpu_execute(bh_ir* bhir)
{
    bh_error res = BH_SUCCESS;
    bh_dag* root = &bhir->dag_list[0];  // Start at the root DAG

    for(bh_intp i=0; i<root->nnode; ++i) {
        bh_intp node = root->node_map[i];
        if (node>0) {
            cout << "Encountered an instruction in the root-dag." << endl;
            return BH_ERROR;
        }
        node = -1*node-1; // Compute the node-index

        //
        // We are now looking at a graph in which we hope that all nodes are instructions
        // we map this to a block in a slightly different format than a list of instructions
        block_t block;
        compose(&block, bhir, &bhir->dag_list[node]);

        //
        // We start by creating a symbol
        if (!symbolize(block, jit_optimize)) {
            cout << "FAILED CREATING SYMBOL" << endl;
            return BH_ERROR;
        }

        cout << block_text(&block) << endl;

        /* TODO: fix this
        // Lets check if it is a known extension method
        {
            map<bh_opcode,bh_extmethod_impl>::iterator ext;
            ext = extmethod_op2impl.find(instr->opcode);
            if (ext != extmethod_op2impl.end()) {
                bh_extmethod_impl extmethod = ext->second;
                return extmethod(instr, NULL);
            }
        }
        */
        //
        // JIT-compile the block if enabled
        //
        if (jit_enabled && \
            (block.symbol!="") && \
            (!target->symbol_ready(block.symbol))) {   
                                                        // Specialize sourcecode
            string sourcecode = specialize(block, jit_optimize);   
            if (jit_dumpsrc==1) {                       // Dump sourcecode to file
                target->src_to_file(
                    block.symbol,
                    sourcecode.c_str(),
                    sourcecode.size()
                );
            }                                           // Send to compiler
            target->compile(block.symbol, sourcecode.c_str(), sourcecode.size());
        }

        //
        // Load the compiled code
        //
        if ((block.symbol!="") && \
            (!target->symbol_ready(block.symbol)) && \
            (!target->load(block.symbol))) {// Need but cannot load

            if (jit_optimize) {                             // Unoptimized fallback
                symbolize(block, false);
                if ((block.symbol!="") && \
                    (!target->symbol_ready(block.symbol)) && \
                    (!target->load(block.symbol))) {        // Fail
                    return BH_ERROR;
                }
            } else {
                return BH_ERROR;
            }
        }

        //
        // Allocate memory for output
        //
        for(int i=0; i<block.length; ++i) {
            res = bh_vcache_malloc(block.instr[i]);
            if (BH_SUCCESS != res) {
                fprintf(stderr, "Unhandled error returned by bh_vcache_malloc() "
                                "called from bh_ve_cpu_execute()\n");
                return res;
            }
        }

        //
        // Execute block handling array operations.
        // 
        if ((block.omask & (BUILTIN_ARRAY_OPS)) > 0) {
            if (BH_SUCCESS != res) {
                fprintf(stderr, "Unhandled error returned by dispatch_block "
                                "called from bh_ve_cpu_execute(...)\n");
                return res;
            }
            target->funcs[block.symbol](block.scope);
        }

        //
        // De-Allocate operand memory
        for(int i=0; i<block.length; ++i) {
            if (block.instr[i]->opcode == BH_FREE) {
                res = bh_vcache_free(block.instr[i]);
                if (BH_SUCCESS != res) {
                    fprintf(stderr, "Unhandled error returned by bh_vcache_free(...) "
                                    "called from bh_ve_cpu_execute)\n");
                    return res;
                }
            }
        }

    }
    return res;
}

/* Component interface: shutdown (see bh_component.h) */
bh_error bh_ve_cpu_shutdown(void)
{
    if (vcache_size>0) {
        bh_vcache_clear();  // De-allocate the malloc-cache
        bh_vcache_delete();
    }

    delete target;          // De-allocate code-generator

    #ifdef PROFILE
    bh_uint64 sum = 0;
    for(size_t i=0; i<BH_NO_OPCODES; ++i) {
        if (times[i]>0) {
            sum += times[i];
            printf(
                "%s, %ld, %f\n",
                bh_opcode_text(i), calls[i], (times[i]/1000000.0)
            );
        }
    }
    if (calls[BH_NO_OPCODES]>0) {
        sum += times[BH_NO_OPCODES];
        printf(
            "%s, %ld, %f\n",
            "Memory", calls[BH_NO_OPCODES], (times[BH_NO_OPCODES]/1000000.0)
        );
    }
    if (calls[BH_NO_OPCODES+1]>0) {
        sum += times[BH_NO_OPCODES+1];
        printf(
            "%s, %ld, %f\n",
            "Kernels", calls[BH_NO_OPCODES+1], (times[BH_NO_OPCODES+1]/1000000.0)
        );
    }
    printf("TOTAL, %f\n", sum/1000000.0);
    #endif

    bh_component_destroy(&myself);

    return BH_SUCCESS;
}

/* Component interface: extmethod (see bh_component.h) */
bh_error bh_ve_cpu_extmethod(const char *name, bh_opcode opcode)
{
    bh_extmethod_impl extmethod;
    bh_error err = bh_component_extmethod(&myself, name, &extmethod);
    if(err != BH_SUCCESS)
        return err;

    if(extmethod_op2impl.find(opcode) != extmethod_op2impl.end())
    {
        printf("[CPU-VE] Warning, multiple registrations of the same"
               "extension method '%s' (opcode: %d)\n", name, (int)opcode);
    }
    extmethod_op2impl[opcode] = extmethod;
    return BH_SUCCESS;
}

