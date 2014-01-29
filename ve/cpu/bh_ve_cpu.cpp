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

//
// NOTE: Changes to bk_kernel_args_t must be 
//       replicated to "templates/kernel.tpl".
//
// TODO: Change this structure....
//
typedef struct bh_kernel_args { 
    int nargs;                  

    void*    data[30];
    int64_t  nelem[30];
    int64_t  ndim[30];
    int64_t  start[30];
    int64_t* shape[30];
    int64_t* stride[30];
} bh_kernel_args_t;

typedef struct bh_kernel {
    int ninstr;
    int nnonsys;

    bh_instruction* instr[10];
    int tsig[10];
    int lmask[10];
    int64_t ndim[10];

    bh_kernel_args_t args;

    string symbol;
} bh_kernel_t;

#include "compiler.cpp"
#include "specializer.cpp"

process* target;

/**
 *  Dispatches / executes a single instruction from a kernel.
 *
 *  Note: System opcodes (including BH_FREE) are ignored.
 *
 *  @param kernel Pointer to the kernel.
 *  @param idx Index of the instruction to execute.
 *  @returns bh_error BH_SUCCESS or BH_ERROR
static bh_error dispatch_si(bh_kernel_t* kernel, int idx)
{
    bh_instruction* instr = kernel->instr[idx];
    int lmask = kernel->lmask[idx];
    bh_error res = BH_ERROR;
    func f = target->funcs[kernel->symbol];

    bh_kernel_operand_t operands;

    switch (instr->opcode) {    // OPCODE_SWITCH    - DISPATCH

        case BH_NONE:               // NOOP.
        case BH_DISCARD:
        case BH_SYNC:
        case BH_FREE:
            res = BH_SUCCESS;
            break;

        case BH_RANDOM:
            f(0,
                bh_base_array(&instr->operand[0])->nelem,
                instr->constant.value.r123.start,
                instr->constant.value.r123.key,
                bh_base_array(&instr->operand[0])->data
            );
            res = BH_SUCCESS;
            break;

        case BH_RANGE:
            f(0,
                bh_base_array(&instr->operand[0])->nelem,
                bh_base_array(&instr->operand[0])->data
            );
            res = BH_SUCCESS;
            break;

        case BH_ADD_ACCUMULATE:                 // Scan
        case BH_MULTIPLY_ACCUMULATE:

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

            f(0,
                bh_base_array(&instr->operand[0])->data,
                instr->operand[0].start,
                instr->operand[0].stride,
                instr->operand[0].shape,
                instr->operand[0].ndim,

                bh_base_array(&instr->operand[1])->data,
                instr->operand[1].start,
                instr->operand[1].stride,
                instr->operand[1].shape,
                instr->operand[1].ndim,

                &(instr->constant.value)
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

            if ((lmask & A2_CONSTANT) == A2_CONSTANT) {         // DDC
                f(0,
                    instr->operand[0].shape,
                    instr->operand[0].ndim,

                    bh_base_array(&instr->operand[0])->data,
                    instr->operand[0].start,
                    instr->operand[0].stride,

                    bh_base_array(&instr->operand[1])->data,
                    instr->operand[1].start,
                    instr->operand[1].stride,

                    &(instr->constant.value)
                );
            } else if ((lmask & A1_CONSTANT) == A1_CONSTANT) { // DCD
                f(0,
                    instr->operand[0].shape,
                    instr->operand[0].ndim,

                    bh_base_array(&instr->operand[0])->data,
                    instr->operand[0].start,
                    instr->operand[0].stride,

                    &(instr->constant.value),

                    bh_base_array(&instr->operand[2])->data,
                    instr->operand[2].start,
                    instr->operand[2].stride
                );
            } else {                                        // DDD
                f(0,
                    instr->operand[0].shape,
                    instr->operand[0].ndim,

                    bh_base_array(&instr->operand[0])->data,
                    instr->operand[0].start,
                    instr->operand[0].stride,

                    bh_base_array(&instr->operand[1])->data,
                    instr->operand[1].start,
                    instr->operand[1].stride,

                    bh_base_array(&instr->operand[2])->data,
                    instr->operand[2].start,
                    instr->operand[2].stride
                );
            }

            res = BH_SUCCESS;
            break;

        case BH_REAL:
        case BH_IMAG:
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

            if ((lmask & A1_CONSTANT) == A1_CONSTANT) {
                f(0,
                    instr->operand[0].shape,
                    instr->operand[0].ndim,

                    bh_base_array(&instr->operand[0])->data,
                    instr->operand[0].start,
                    instr->operand[0].stride,

                    &(instr->constant.value)
                );
            } else {
                f(0,
                    instr->operand[0].shape,
                    instr->operand[0].ndim,

                    bh_base_array(&instr->operand[0])->data,
                    instr->operand[0].start,
                    instr->operand[0].stride,

                    bh_base_array(&instr->operand[1])->data,
                    instr->operand[1].start,
                    instr->operand[1].stride
                );
            }
            res = BH_SUCCESS;
            break;

        default:                            // Shit hit the fan
            res = BH_ERROR;
            printf("cpu-exec: Err=[Unsupported ufunc] {\n");
            bh_pprint_instr(instr);
            printf("}\n");
    }

    return res;
}

*/

/// I WILL DO SOME SERIOUS AMON TOBIN ON THIS CODE!!!
//
/**
 *  Pack kernel arguments and execute kernel-function.
 *
 *  Contract: Do not call this function the kernel.nnonsys == 0.
 */
static bh_error pack_arguments(bh_kernel_t* kernel)
{
    //
    // Setup arguments
    //
    int nargs=0;
    for (int i=0; i<kernel->ninstr; ++i) {
        bh_instruction* instr = kernel->instr[i];

        //
        // Do not pack operands of system opcodes.
        //
        if ((instr->opcode >= BH_DISCARD) && (instr->opcode <= BH_SYNC)) {
            continue;
        }

        //
        // The layoutmask is used to determine how to pack arguments.
        //
        int lmask = kernel->lmask[i];

        switch (instr->opcode) {    // [OPCODE_SWITCH]

            case BH_RANDOM:
                kernel->args.data[nargs++] = bh_base_array(&instr->operand[0])->data;
                kernel->args.data[nargs++] = &(instr->constant.value.r123.start);
                kernel->args.data[nargs++] = &(instr->constant.value.r123.key);
                break;

            case BH_RANGE:
                kernel->args.data[nargs++] = bh_base_array(&instr->operand[0])->data;
                break;

            case BH_ADD_ACCUMULATE:                 // Scan
            case BH_MULTIPLY_ACCUMULATE:

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

                kernel->args.data[nargs++] = bh_base_array(&instr->operand[0])->data;
                kernel->args.data[nargs++] = bh_base_array(&instr->operand[1])->data;
                kernel->args.data[nargs++] = &(instr->constant.value);
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

                kernel->args.data[nargs++] = bh_base_array(&instr->operand[0])->data;
                if ((lmask & A2_CONSTANT) == A2_CONSTANT) {         // CCK
                    kernel->args.data[nargs++] = bh_base_array(&instr->operand[1])->data;
                    kernel->args.data[nargs++] = &(instr->constant.value);
                } else if ((lmask & A1_CONSTANT) == A1_CONSTANT) {  // CKC
                    kernel->args.data[nargs++] = &(instr->constant.value);
                    kernel->args.data[nargs++] = bh_base_array(&instr->operand[2])->data;
                } else {                                            // CCC
                    kernel->args.data[nargs++] = bh_base_array(&instr->operand[1])->data;
                    kernel->args.data[nargs++] = bh_base_array(&instr->operand[2])->data;
                }

                break;

            case BH_REAL:
            case BH_IMAG:
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

                // Output is always an array...
                kernel->args.data[nargs]     = bh_base_array(&instr->operand[0])->data;
                kernel->args.nelem[nargs]    = bh_base_array(&instr->operand[0])->nelem;
                kernel->args.ndim[nargs]     = instr->operand[0].ndim;
                kernel->args.start[nargs]    = instr->operand[0].start;
                kernel->args.shape[nargs]    = instr->operand[0].shape;
                kernel->args.stride[nargs++] = instr->operand[0].stride;

                // Input might be a constant
                if ((lmask & A1_CONSTANT) == A1_CONSTANT) {
                    kernel->args.data[nargs++] = &(instr->constant.value);
                } else {
                    kernel->args.data[nargs]     = bh_base_array(&instr->operand[1])->data;
                    kernel->args.nelem[nargs]    = bh_base_array(&instr->operand[1])->nelem;
                    kernel->args.ndim[nargs]     = instr->operand[1].ndim;
                    kernel->args.start[nargs]    = instr->operand[1].start;
                    kernel->args.shape[nargs]    = instr->operand[1].shape;
                    kernel->args.stride[nargs++] = instr->operand[1].stride;
                }

                break;

            default:
                printf("cpu_pack_arguments: Err=[Unsupported instruction] {\n");
                bh_pprint_instr(instr);
                printf("}\n");
                return BH_ERROR;
        }
    }

    //
    // Update the argument count for the kernel
    //
    kernel->args.nargs = nargs; 
        
    return BH_SUCCESS;
}

// Execute a kernel
static bh_error exec_kernel(bh_instruction *instr)
{
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

    bh_kernel_t kernel;
    kernel.instr[0] = instr;
    kernel.ninstr = 1;

    if (!symbolize(kernel, jit_optimize)) {
        return BH_ERROR;
    }

    //
    // JIT-compile the kernel if enabled
    //
    if (jit_enabled && \
        (kernel.symbol!="") && \
        (!target->symbol_ready(kernel.symbol))) {   
                                                    // Specialize sourcecode
        string sourcecode = specialize(kernel, jit_optimize);   
        if (jit_dumpsrc==1) {                       // Dump sourcecode to file
            target->src_to_file(
                kernel.symbol,
                sourcecode.c_str(),
                sourcecode.size()
            );
        }                                           // Send to compiler
        target->compile(kernel.symbol, sourcecode.c_str(), sourcecode.size());
    }

    //
    // Load the compiled code
    //
    if ((kernel.symbol!="") && \
        (!target->symbol_ready(kernel.symbol)) && \
        (!target->load(kernel.symbol, kernel.symbol))) {// Need but cannot load

        if (jit_optimize) {                             // Unoptimized fallback
            symbolize(kernel, false);
            if ((kernel.symbol!="") && \
                (!target->symbol_ready(kernel.symbol)) && \
                (!target->load(kernel.symbol, kernel.symbol))) {        // Fail
                return BH_ERROR;
            }
        } else {
            return BH_ERROR;
        }
    }

    //
    // Allocate memory for output
    //
    for(int i=0; i<kernel.ninstr; ++i) {
        res = bh_vcache_malloc(kernel.instr[i]);
        if (BH_SUCCESS != res) {
            fprintf(stderr, "Unhandled error returned by bh_vcache_malloc() "
                            "called from bh_ve_cpu_exec_kernel()\n");
            return res;
        }
    }
    
    //
    // Execute kernel handling array operations.
    // 
    if (kernel.nnonsys>0) {
        res = pack_arguments(&kernel);
        if (BH_SUCCESS != res) {
            fprintf(stderr, "Unhandled error returned by dispatch_kernel "
                            "called from bh_ve_cpu_exec_kernel(...)\n");
            return res;
        }
        target->funcs[kernel.symbol](&kernel.args);
    }

    //
    // De-Allocate memory
    //
    for(int i=0; i<kernel.ninstr; ++i) {
        if (kernel.instr[i]->opcode == BH_FREE) {
            res = bh_vcache_free(kernel.instr[i]);
            if (BH_SUCCESS != res) {
                fprintf(stderr, "Unhandled error returned by bh_vcache_free(...) "
                                "called from bh_ve_cpu_exec_kernel)\n");
                return res;
            }
        }
    }

    return res;
}

//
//  Methods below implement the component interface
//

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
    return bh_ir_map_instr(bhir, &bhir->dag_list[0], &exec_kernel);
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

