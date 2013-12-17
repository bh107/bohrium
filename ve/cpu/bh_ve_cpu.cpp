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

typedef struct bh_sij {
    bh_instruction *instr;  // Pointer to instruction
    int64_t ndims;          // Number of dimensions
    int lmask;              // Layout mask
    int tsig;               // Type signature

    string symbol;          // String representation
} bh_sij_t;                 // Encapsulation of single-instruction(expression)-jit

#include "compiler.cpp"
#include "specializer.cpp"

process* target;

// Execute a single instruction
static bh_error exec(bh_instruction *instr)
{
    bh_sij_t sij;
    bh_error res = BH_SUCCESS;

    // Lets check if it is a known extension method
    {
        map<bh_opcode,bh_extmethod_impl>::iterator ext;
        ext = extmethod_op2impl.find(instr->opcode);
        if (ext != extmethod_op2impl.end()) {
            bh_extmethod_impl extmethod = ext->second;
            return extmethod(instr, NULL);
        }
    }

    if (!symbolize(instr, sij, jit_optimize)) {     
        return BH_ERROR;
    }

    if (jit_enabled && \
        (sij.symbol!="") && \
        (!target->symbol_ready(sij.symbol))) {      // JIT-compile the function

        string sourcecode = specialize(sij, jit_optimize);   // Specialize sourcecode
        if (jit_dumpsrc==1) {                       // Dump sourcecode to file
            target->src_to_file(sij.symbol, sourcecode.c_str(), sourcecode.size());
        }                                           // Send to code generator
        target->compile(sij.symbol, sourcecode.c_str(), sourcecode.size());
    }

    if ((sij.symbol!="") && \
        (!target->symbol_ready(sij.symbol)) && \
        (!target->load(sij.symbol, sij.symbol))) {  // Need but cannot load

        if (jit_optimize) {                         // Try unoptimized symbol
            symbolize(instr, sij, false);
            if ((sij.symbol!="") && \
                (!target->symbol_ready(sij.symbol)) && \
                (!target->load(sij.symbol, sij.symbol))) {  // Still cannot load
                return BH_ERROR;
            }
        } else {
            return BH_ERROR;
        }
    }

    res = bh_vcache_malloc(sij.instr);              // Allocate memory for operands
    if (BH_SUCCESS != res) {
        fprintf(stderr, "Unhandled error returned by bh_vcache_malloc() "
                        "called from bh_ve_cpu_execute()\n");
        return res;
    }

    switch (sij.instr->opcode) {    // OPCODE_SWITCH    - DISPATCH

        case BH_NONE:               // NOOP.
        case BH_DISCARD:
        case BH_SYNC:
            res = BH_SUCCESS;
            break;

        case BH_FREE:                           // Store data-pointer in malloc-cache
            res = bh_vcache_free(sij.instr);
            break;

        case BH_RANDOM:
            target->funcs[sij.symbol](0,
                bh_base_array(&sij.instr->operand[0])->data,
                bh_base_array(&sij.instr->operand[0])->nelem,
                sij.instr->constant.value.r123.start,
                sij.instr->constant.value.r123.key
            );
            res = BH_SUCCESS;
            break;

        case BH_RANGE:
            target->funcs[sij.symbol](0,
                bh_base_array(&sij.instr->operand[0])->data,
                bh_base_array(&sij.instr->operand[0])->nelem
            );
            res = BH_SUCCESS;
            break;

        case BH_ADD_REDUCE:                     // Partial Reductions
        case BH_MULTIPLY_REDUCE:
        case BH_MINIMUM_REDUCE:
        case BH_MAXIMUM_REDUCE:
        case BH_LOGICAL_AND_REDUCE:
        case BH_BITWISE_AND_REDUCE:
        case BH_LOGICAL_OR_REDUCE:
        case BH_LOGICAL_XOR_REDUCE:
        case BH_BITWISE_OR_REDUCE:
        case BH_BITWISE_XOR_REDUCE:

            target->funcs[sij.symbol](0,
                bh_base_array(&sij.instr->operand[0])->data,
                sij.instr->operand[0].start,
                sij.instr->operand[0].stride,
                sij.instr->operand[0].shape,
                sij.instr->operand[0].ndim,

                bh_base_array(&sij.instr->operand[1])->data,
                sij.instr->operand[1].start,
                sij.instr->operand[1].stride,
                sij.instr->operand[1].shape,
                sij.instr->operand[1].ndim,

                sij.instr->constant.value
            );
            res = BH_SUCCESS;
            break;

        case BH_ADD:
        case BH_SUBTRACT:
        case BH_MULTIPLY:
        case BH_DIVIDE:
        case BH_POWER:
        case BH_GREATER:
        case BH_GREATER_EQUAL:
        case BH_LESS:
        case BH_LESS_EQUAL:
        case BH_EQUAL:
        case BH_NOT_EQUAL:
        case BH_LOGICAL_AND:
        case BH_LOGICAL_OR:
        case BH_LOGICAL_XOR:
        case BH_MAXIMUM:
        case BH_MINIMUM:
        case BH_BITWISE_AND:
        case BH_BITWISE_OR:
        case BH_BITWISE_XOR:
        case BH_LEFT_SHIFT:
        case BH_RIGHT_SHIFT:
        case BH_ARCTAN2:
        case BH_MOD:

            if ((sij.lmask & A2_CONSTANT) == A2_CONSTANT) {         // DDC
                target->funcs[sij.symbol](0,
                    bh_base_array(&sij.instr->operand[0])->data,
                    sij.instr->operand[0].start,
                    sij.instr->operand[0].stride,

                    bh_base_array(&sij.instr->operand[1])->data,
                    sij.instr->operand[1].start,
                    sij.instr->operand[1].stride,

                    &(sij.instr->constant.value),

                    sij.instr->operand[0].shape,
                    sij.instr->operand[0].ndim
                );
            } else if ((sij.lmask & A1_CONSTANT) == A1_CONSTANT) { // DCD
                target->funcs[sij.symbol](0,
                    bh_base_array(&sij.instr->operand[0])->data,
                    sij.instr->operand[0].start,
                    sij.instr->operand[0].stride,

                    &(sij.instr->constant.value),

                    bh_base_array(&sij.instr->operand[2])->data,
                    sij.instr->operand[2].start,
                    sij.instr->operand[2].stride,

                    sij.instr->operand[0].shape,
                    sij.instr->operand[0].ndim
                );
            } else {                                        // DDD
                target->funcs[sij.symbol](0,
                    bh_base_array(&sij.instr->operand[0])->data,
                    sij.instr->operand[0].start,
                    sij.instr->operand[0].stride,

                    bh_base_array(&sij.instr->operand[1])->data,
                    sij.instr->operand[1].start,
                    sij.instr->operand[1].stride,

                    bh_base_array(&sij.instr->operand[2])->data,
                    sij.instr->operand[2].start,
                    sij.instr->operand[2].stride,

                    sij.instr->operand[0].shape,
                    sij.instr->operand[0].ndim
                );
            }

            res = BH_SUCCESS;
            break;

        case BH_ABSOLUTE:
        case BH_LOGICAL_NOT:
        case BH_INVERT:
        case BH_COS:
        case BH_SIN:
        case BH_TAN:
        case BH_COSH:
        case BH_SINH:
        case BH_TANH:
        case BH_ARCSIN:
        case BH_ARCCOS:
        case BH_ARCTAN:
        case BH_ARCSINH:
        case BH_ARCCOSH:
        case BH_ARCTANH:
        case BH_EXP:
        case BH_EXP2:
        case BH_EXPM1:
        case BH_LOG:
        case BH_LOG2:
        case BH_LOG10:
        case BH_LOG1P:
        case BH_SQRT:
        case BH_CEIL:
        case BH_TRUNC:
        case BH_FLOOR:
        case BH_RINT:
        case BH_ISNAN:
        case BH_ISINF:
        case BH_IDENTITY:

            if ((sij.lmask & A1_CONSTANT) == A1_CONSTANT) {
                target->funcs[sij.symbol](0,
                    bh_base_array(&sij.instr->operand[0])->data,
                    sij.instr->operand[0].start,
                    sij.instr->operand[0].stride,

                    &(sij.instr->constant.value),

                    sij.instr->operand[0].shape,
                    sij.instr->operand[0].ndim
                );
            } else {
                target->funcs[sij.symbol](0,
                    bh_base_array(&sij.instr->operand[0])->data,
                    sij.instr->operand[0].start,
                    sij.instr->operand[0].stride,

                    bh_base_array(&sij.instr->operand[1])->data,
                    sij.instr->operand[1].start,
                    sij.instr->operand[1].stride,

                    sij.instr->operand[0].shape,
                    sij.instr->operand[0].ndim
                );
            }
            res = BH_SUCCESS;
            break;

        default:                            // Shit hit the fan
            res = BH_ERROR;
            printf("cpu: Err=[Unsupported ufunc...\n");
    }
    return res;
}

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
    target = new process(compiler_cmd, object_path, kernel_path, jit_preload);
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
    // Execute one instruction at a time starting at the root DAG.
    return bh_ir_map_instr(bhir, &bhir->dag_list[0], &exec);
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

