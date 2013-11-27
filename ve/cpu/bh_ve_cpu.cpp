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
#include "compiler.cpp"

using namespace std;

// Execution Profile

#ifdef PROFILE
static bh_uint64 times[BH_NO_OPCODES+2]; // opcodes and: +1=malloc, +2=kernel
static bh_uint64 calls[BH_NO_OPCODES+2];
#endif

static bh_component *myself = NULL;
static bh_userfunc_impl random_impl = NULL;
static bh_intp random_impl_id = 0;
static bh_userfunc_impl matmul_impl = NULL;
static bh_intp matmul_impl_id = 0;
static bh_userfunc_impl nselect_impl = NULL;
static bh_intp nselect_impl_id = 0;
static bh_userfunc_impl visualizer_impl = NULL;
static bh_intp visualizer_impl_id = 0;

static bh_intp vcache_size   = 10;
static bh_intp do_fuse = 1;
static bh_intp do_jit  = 1;
static bh_intp dump_src = 0;

static char* compiler_cmd;   // cpu Arguments
static char* kernel_path;
static char* object_path;
static char* template_path;

process* target;

typedef struct bh_sij {
    bh_instruction *instr;  // Pointer to instruction
    int64_t ndims;          // Number of dimensions
    int lmask;              // Layout mask
    int tsig;               // Type signature

    string symbol;     // String representation
} bh_sij_t;                 // Encapsulation of single-instruction(expression)-jit

void bh_string_option(char *&option, const char *env_name, const char *conf_name)
{
    option = getenv(env_name);           // For the compiler
    if (NULL==option) {
        option = bh_component_config_lookup(myself, conf_name);
    }
    char err_msg[100];

    if (!option) {
        sprintf(err_msg, "cpu-ve: String is not set; option (%s).\n", conf_name);
        throw runtime_error(err_msg);
    }
}

void bh_path_option(char *&option, const char *env_name, const char *conf_name)
{
    option = getenv(env_name);           // For the compiler
    if (NULL==option) {
        option = bh_component_config_lookup(myself, conf_name);
    }
    char err_msg[100];

    if (!option) {
        sprintf(err_msg, "cpu-ve: Path is not set; option (%s).\n", conf_name);
        throw runtime_error(err_msg);
    }
    if (0 != access(option, F_OK)) {
        if (ENOENT == errno) {
            sprintf(err_msg, "cpu-ve: Path does not exist; path (%s).\n", option);
        } else if (ENOTDIR == errno) {
            sprintf(err_msg, "cpu-ve: Path is not a directory; path (%s).\n", option);
        } else {
            sprintf(err_msg, "cpu-ve: Path is broken somehow; path (%s).\n", option);
        }
        throw runtime_error(err_msg);
    }
}

bh_error bh_ve_cpu_init(bh_component *self)
{
    myself = self;
    char *env;

    env = getenv("BH_CORE_VCACHE_SIZE");      // Override block_size from environment-variable.
    if (NULL != env) {
        vcache_size = atoi(env);
    }
    if (0 > vcache_size) {                          // Verify it
        fprintf(stderr, "BH_CORE_VCACHE_SIZE (%ld) should be greater than zero!\n", (long int)vcache_size);
        return BH_ERROR;
    }

    env = getenv("BH_VE_CPU_DOFUSE");
    if (NULL != env) {
        do_fuse = atoi(env);
    }
    if (!((0==do_fuse) || (1==do_fuse))) {
        fprintf(stderr, "BH_VE_CPU_DOFUSE (%ld) should 0 or 1.\n", (long int)do_fuse);
        return BH_ERROR;
    }

    env = getenv("BH_VE_CPU_DUMPSRC");
    if (NULL != env) {
        dump_src = atoi(env);
    }
    if (!((0==dump_src) || (1==dump_src))) {
         fprintf(stderr, "BH_VE_CPU_DUMPSRC (%ld) should 0 or 1.\n", (long int)dump_src);
        return BH_ERROR;
    }

    bh_vcache_init(vcache_size);

    // CPU Arguments
    bh_path_option(     kernel_path,    "BH_VE_CPU_KERNEL_PATH",   "kernel_path");
    bh_path_option(     object_path,    "BH_VE_CPU_OBJECT_PATH",   "object_path");
    bh_path_option(     template_path,  "BH_VE_CPU_TEMPLATE_PATH", "template_path");
    bh_string_option(   compiler_cmd,   "BH_VE_CPU_COMPILER",      "compiler_cmd");

    target = new process(compiler_cmd, object_path, kernel_path, true);

    #ifdef PROFILE
    memset(&times, 0, sizeof(bh_uint64)*(BH_NO_OPCODES+2));
    memset(&calls, 0, sizeof(bh_uint64)*(BH_NO_OPCODES+2));
    #endif

    return BH_SUCCESS;
}

void symbolize(bh_instruction *instr, bh_sij_t &sij) {

    char symbol_c[500];             // String representation buffers
    char dims_str[10];

    sij.instr = instr;
    switch (sij.instr->opcode) {                    // [OPCODE_SWITCH]

        case BH_NONE:                           // System opcodes
        case BH_DISCARD:
        case BH_SYNC:
        case BH_FREE:
        case BH_USERFUNC:                       // Extensions
            break;

        case BH_ADD_REDUCE:                             // Reductions
        case BH_MULTIPLY_REDUCE:
        case BH_MINIMUM_REDUCE:
        case BH_MAXIMUM_REDUCE:
        case BH_LOGICAL_AND_REDUCE:
        case BH_BITWISE_AND_REDUCE:
        case BH_LOGICAL_OR_REDUCE:
        case BH_LOGICAL_XOR_REDUCE:
        case BH_BITWISE_OR_REDUCE:
        case BH_BITWISE_XOR_REDUCE:
            sij.ndims = sij.instr->operand[1].ndim;     // Dimensions
            sij.lmask = bh_layoutmask(sij.instr);       // Layout mask
            sij.tsig  = bh_typesig(sij.instr);          // Type signature

            if (sij.ndims <= 3) {                       // String representation
                sprintf(dims_str, "%ldD", sij.ndims);
            } else {
                sprintf(dims_str, "ND");
            }
            sprintf(symbol_c, "%s_%s_%s_%s",
                bh_opcode_text(sij.instr->opcode),
                dims_str,
                bh_layoutmask_to_shorthand(sij.lmask),
                bh_typesig_to_shorthand(sij.tsig)
            );

            sij.symbol = string(symbol_c);
            break;

        case BH_RANGE:

            sij.ndims = sij.instr->operand[0].ndim;     // Dimensions
            sij.lmask = bh_layoutmask(sij.instr);       // Layout mask
            sij.tsig  = bh_typesig(sij.instr);          // Type signature

            sprintf(symbol_c, "%s_ND_%s_%s",
                bh_opcode_text(sij.instr->opcode),
                bh_layoutmask_to_shorthand(sij.lmask),
                bh_typesig_to_shorthand(sij.tsig)
            );

            sij.symbol = string(symbol_c);
            break;

        default:                                        // Built-in
            
            sij.ndims = sij.instr->operand[0].ndim;     // Dimensions
            sij.lmask = bh_layoutmask(sij.instr);       // Layout mask
            sij.tsig  = bh_typesig(sij.instr);          // Type signature

            if (sij.ndims <= 3) {                       // String representation
                sprintf(dims_str, "%ldD", sij.ndims);
            } else {
                sprintf(dims_str, "ND");
            }
            sprintf(symbol_c, "%s_%s_%s_%s",
                bh_opcode_text(sij.instr->opcode),
                dims_str,
                bh_layoutmask_to_shorthand(sij.lmask),
                bh_typesig_to_shorthand(sij.tsig)
            );

            sij.symbol = string(symbol_c);
            break;
    }
}

string specialize(bh_sij_t &sij) {

    char template_fn[500];   // NOTE: constants like these are often traumatizing!

    bool cres = false;

    ctemplate::TemplateDictionary dict("codegen");
    dict.ShowSection("include");
    //dict.ShowSection("license");

    switch (sij.instr->opcode) {                    // OPCODE_SWITCH

        case BH_RANGE:
            dict.SetValue("OPERATOR", bhopcode_to_cexpr(sij.instr->opcode));
            dict.SetValue("SYMBOL", sij.symbol);
            dict.SetValue("TYPE_A0", bhtype_to_ctype(sij.instr->operand[0].base->type));
            sprintf(template_fn, "%s/range.tpl", template_path);

            cres = true;
            break;

        case BH_ADD_REDUCE:
        case BH_MULTIPLY_REDUCE:
        case BH_MINIMUM_REDUCE:
        case BH_MAXIMUM_REDUCE:
        case BH_LOGICAL_AND_REDUCE:
        case BH_BITWISE_AND_REDUCE:
        case BH_LOGICAL_OR_REDUCE:
        case BH_LOGICAL_XOR_REDUCE:
        case BH_BITWISE_OR_REDUCE:
        case BH_BITWISE_XOR_REDUCE:

            dict.SetValue("OPERATOR", bhopcode_to_cexpr(sij.instr->opcode));
            dict.SetValue("SYMBOL", sij.symbol);
            dict.SetValue("TYPE_A0", bhtype_to_ctype(sij.instr->operand[0].base->type));
            dict.SetValue("TYPE_A1", bhtype_to_ctype(sij.instr->operand[1].base->type));

            if (sij.ndims <= 3) {
                sprintf(template_fn, "%s/reduction.%ldd.tpl", template_path, sij.ndims);
            } else {
                sprintf(template_fn, "%s/reduction.nd.tpl", template_path);
            }

            cres = true;
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
        case BH_RANDOM:

            dict.SetValue("OPERATOR", bhopcode_to_cexpr(sij.instr->opcode));
            if ((sij.lmask & A2_CONSTANT) == A2_CONSTANT) {
                dict.SetValue("SYMBOL", sij.symbol);
                dict.SetValue("TYPE_A0", bhtype_to_ctype(sij.instr->operand[0].base->type));
                dict.SetValue("TYPE_A1", bhtype_to_ctype(sij.instr->operand[1].base->type));
                dict.SetValue("TYPE_A2", bhtype_to_ctype(sij.instr->constant.type));
                dict.ShowSection("a1_dense");
                dict.ShowSection("a2_scalar");
            } else if ((sij.lmask & A1_CONSTANT) == A1_CONSTANT) {
                dict.SetValue("SYMBOL", sij.symbol);
                dict.SetValue("TYPE_A0", bhtype_to_ctype(sij.instr->operand[0].base->type));
                dict.SetValue("TYPE_A1", bhtype_to_ctype(sij.instr->constant.type));
                dict.SetValue("TYPE_A2", bhtype_to_ctype(sij.instr->operand[2].base->type));
                dict.ShowSection("a1_scalar");
                dict.ShowSection("a2_dense");
            } else {
                dict.SetValue("SYMBOL", sij.symbol);
                dict.SetValue("TYPE_A0", bhtype_to_ctype(sij.instr->operand[0].base->type));
                dict.SetValue("TYPE_A1", bhtype_to_ctype(sij.instr->operand[1].base->type));
                dict.SetValue("TYPE_A2", bhtype_to_ctype(sij.instr->operand[2].base->type));
                dict.ShowSection("a1_dense");
                dict.ShowSection("a2_dense");

            }
            if ((sij.lmask == (A0_DENSE + A1_DENSE    + A2_DENSE)) || \
                (sij.lmask == (A0_DENSE + A1_CONSTANT + A2_DENSE)) || \
                (sij.lmask == (A0_DENSE + A1_DENSE    + A2_CONSTANT))) {
                sprintf(template_fn, "%s/traverse.nd.ddd.tpl", template_path);
            } else {
                if (sij.ndims<=3) {
                    sprintf(template_fn, "%s/traverse.%ldd.tpl", template_path, sij.ndims);
                } else {
                    sprintf(template_fn, "%s/traverse.nd.tpl", template_path);
                }
            }

            cres = true;
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

            dict.SetValue("OPERATOR", bhopcode_to_cexpr(sij.instr->opcode));
            if ((sij.lmask & A1_CONSTANT) == A1_CONSTANT) {
                dict.SetValue("SYMBOL", sij.symbol);
                dict.SetValue("TYPE_A0", bhtype_to_ctype(sij.instr->operand[0].base->type));
                dict.SetValue("TYPE_A1", bhtype_to_ctype(sij.instr->constant.type));
                dict.ShowSection("a1_scalar");
            } else {
                dict.SetValue("SYMBOL", sij.symbol);
                dict.SetValue("TYPE_A0", bhtype_to_ctype(sij.instr->operand[0].base->type));
                dict.SetValue("TYPE_A1", bhtype_to_ctype(sij.instr->operand[1].base->type));
                dict.ShowSection("a1_dense");
            }

            if ((sij.lmask == (A0_DENSE + A1_DENSE)) || \
                (sij.lmask == (A0_DENSE + A1_CONSTANT))) {
                sprintf(template_fn, "%s/traverse.nd.ddd.tpl", template_path);
            } else {

                if (sij.ndims<=3) {
                    sprintf(template_fn, "%s/traverse.%ldd.tpl", template_path, sij.ndims);
                } else {
                    sprintf(template_fn, "%s/traverse.nd.tpl", template_path);
                }

            }

            cres = true;
            break;

        default:
            printf("cpu-ve: Err=[Unsupported ufunc...]\n");
    }

    if (!cres) {
        throw runtime_error("cpu-ve: Failed specializing code.");
    }

    string sourcecode = "";
    ctemplate::ExpandTemplate(
        template_fn,
        ctemplate::STRIP_BLANK_LINES,
        &dict,
        &sourcecode
    );

    return sourcecode;
}

// Executes a single instruction
static bh_error exec(bh_instruction *instr)
{
    bh_sij_t        sij;
    bh_error res = BH_SUCCESS;

    symbolize(instr, sij);           // Construct symbol
    if (do_jit && (sij.symbol!="") && (!target->symbol_ready(sij.symbol))) {

        string sourcecode = specialize(sij);   // Specialize sourcecode
        if (dump_src==1) {                          // Dump sourcecode to file
            target->src_to_file(sij.symbol, sourcecode.c_str(), sourcecode.size());
        }                                           // Send to code generator
        target->compile(sij.symbol, sourcecode.c_str(), sourcecode.size());
    }

    if ((sij.symbol!="") && !target->load(sij.symbol, sij.symbol)) {// Load
        return BH_ERROR;
    }
    res = bh_vcache_malloc(sij.instr);                          // Allocate memory for operands
    if (BH_SUCCESS != res) {
        fprintf(stderr, "Unhandled error returned by bh_vcache_malloc() "
                        "called from bh_ve_cpu_execute()\n");
        return res;
    }

    switch (sij.instr->opcode) {    // OPCODE_SWITCH

        case BH_NONE:               // NOOP.
        case BH_DISCARD:
        case BH_SYNC:
            res = BH_SUCCESS;
            break;

        case BH_FREE:                           // Store data-pointer in malloc-cache
            res = bh_vcache_free(sij.instr);
            break;

        case BH_USERFUNC:
            if (sij.instr->userfunc->id == matmul_impl_id) {
                res = matmul_impl(sij.instr->userfunc, NULL);
            } else if (sij.instr->userfunc->id == nselect_impl_id) {
                res = nselect_impl(sij.instr->userfunc, NULL);
            } else if (sij.instr->userfunc->id == visualizer_impl_id) {
                res = visualizer_impl(sij.instr->userfunc, NULL);
            } else {                            // Unsupported userfunc
                res = BH_USERFUNC_NOT_SUPPORTED;
            }
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
        case BH_RANDOM:

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

bh_error bh_ve_cpu_execute(bh_ir* bhir)
{
    //Execute one instruction at a time starting at the root DAG.
    return bh_ir_map_instr(bhir, &bhir->dag_list[0], &exec);
}

bh_error bh_ve_cpu_shutdown(void)
{
    if (vcache_size>0) {
        bh_vcache_clear();  // De-allocate the malloc-cache
        bh_vcache_delete();
    }

    delete target;  // De-allocate code-generator

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

    return BH_SUCCESS;
}

bh_error bh_ve_cpu_reg_func(char *fun, bh_intp *id)
{
    if (strcmp("bh_random", fun) == 0) {
    	if (random_impl == NULL) {
            random_impl_id = *id;
            return BH_SUCCESS;
        } else {
        	*id = random_impl_id;
        	return BH_SUCCESS;
        }
    } else if (strcmp("bh_matmul", fun) == 0) {
    	if (matmul_impl == NULL) {
            bh_component_get_func(myself, fun, &matmul_impl);
            if (matmul_impl == NULL) {
                return BH_USERFUNC_NOT_SUPPORTED;
            }

            matmul_impl_id = *id;
            return BH_SUCCESS;
        } else {
        	*id = matmul_impl_id;
        	return BH_SUCCESS;
        }
    }  else if (strcmp("bh_visualizer", fun) == 0) {
    	if (visualizer_impl == NULL) {
            bh_component_get_func(myself, fun, &visualizer_impl);
            if (visualizer_impl == NULL) {
                return BH_USERFUNC_NOT_SUPPORTED;
            }

            visualizer_impl_id = *id;
            return BH_SUCCESS;
        } else {
        	*id = visualizer_impl_id;
        	return BH_SUCCESS;
        }
    } else if(strcmp("bh_nselect", fun) == 0) {
        if (nselect_impl == NULL) {
            bh_component_get_func(myself, fun, &nselect_impl);
            if (nselect_impl == NULL) {
                return BH_USERFUNC_NOT_SUPPORTED;
            }
            nselect_impl_id = *id;
            return BH_SUCCESS;
        } else {
            *id = nselect_impl_id;
            return BH_SUCCESS;
        }
    }

    return BH_USERFUNC_NOT_SUPPORTED;
}

