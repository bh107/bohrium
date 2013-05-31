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
#include <bh.h>
#include "bh_ve_dynamite.h"
#include <bh_vcache.h>
#include <ctemplate/template.h>  
#include "compiler.cpp"
#include <string>
#include <stdexcept>
#include <unistd.h>
#include <errno.h>
#include <inttypes.h>

static bh_component *myself = NULL;
static bh_userfunc_impl random_impl = NULL;
static bh_intp random_impl_id = 0;
static bh_userfunc_impl matmul_impl = NULL;
static bh_intp matmul_impl_id = 0;
static bh_userfunc_impl nselect_impl = NULL;
static bh_intp nselect_impl_id = 0;

static bh_intp vcache_size   = 10;

char* compiler_cmd;   // Dynamite Arguments
char* kernel_path;
char* object_path;
char* snippet_path;

process* target;

void bh_string_option(char *&option, const char *env_name, const char *conf_name)
{
    option = getenv(env_name);           // For the compiler
    if (NULL==option) {
        option = bh_component_config_lookup(myself, conf_name);
    }
    char err_msg[100];

    if (!option) {
        sprintf(err_msg, "Err: String is not set; option (%s).\n", conf_name);
        throw std::runtime_error(err_msg);
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
        sprintf(err_msg, "Err: Path is not set; option (%s).\n", conf_name);
        throw std::runtime_error(err_msg);
    }
    if (0 != access(option, F_OK)) {
        if (ENOENT == errno) {
            sprintf(err_msg, "Err: Path does not exist; path (%s).\n", option);
        } else if (ENOTDIR == errno) {
            sprintf(err_msg, "Err: Path is not a directory; path (%s).\n", option);
        } else {
            sprintf(err_msg, "Err: Path is broken somehow; path (%s).\n", option);
        }
        throw std::runtime_error(err_msg);
    }
}

bh_error bh_ve_dynamite_init(bh_component *self)
{
    myself = self;

    char *env = getenv("BH_CORE_VCACHE_SIZE");      // Override block_size from environment-variable.
    if (NULL != env) {
        vcache_size = atoi(env);
    }
    if (0 >= vcache_size) {                          // Verify it
        fprintf(stderr, "BH_CORE_VCACHE_SIZE (%ld) should be greater than zero!\n", (long int)vcache_size);
        return BH_ERROR;
    }

    bh_vcache_init( vcache_size );

    // DYNAMITE Arguments
    bh_path_option(
        kernel_path,    "BH_VE_DYNAMITE_KERNEL_PATH",   "kernel_path");
    bh_path_option(
        object_path,    "BH_VE_DYNAMITE_OBJECT_PATH",   "object_path");
    bh_path_option(
        snippet_path,   "BH_VE_DYNAMITE_SNIPPET_PATH",  "snippet_path");
    bh_string_option(
        compiler_cmd,   "BH_VE_DYNAMITE_TARGET",        "compiler_cmd");

    target = new process(compiler_cmd, object_path, kernel_path);

    return BH_SUCCESS;
}

bh_error bh_ve_dynamite_execute(bh_intp instruction_count, bh_instruction* instruction_list)
{
    bh_intp count;
    bh_instruction* instr;
    bh_error res = BH_SUCCESS;

    for (count=0; count<instruction_count; count++) {

        ctemplate::TemplateDictionary dict("codegen");
        bh_random_type *random_args;

        bool cres = false;

        std::string sourcecode = "";
        std::string symbol = "";
        int64_t dims = 0;

        char snippet_fn[250];   // NOTE: constants like these are often traumatizing!
        char symbol_c[500];

        instr = &instruction_list[count];

        res = bh_vcache_malloc(instr);              // Allocate memory for operands
        if (BH_SUCCESS != res) {
            printf("Unhandled error returned by bh_vcache_malloc() called from bh_ve_dynamite_execute()\n");
            return res;
        }
                                                    
        switch (instr->opcode) {                    // Dispatch instruction

            case BH_NONE:                           // NOOP.
            case BH_DISCARD:
            case BH_SYNC:
                res = BH_SUCCESS;
                break;
            case BH_FREE:                           // Store data-pointer in malloc-cache
                res = bh_vcache_free(instr);
                break;

            // Extensions (ufuncs)
            case BH_USERFUNC:                    // External libraries

                if(instr->userfunc->id == random_impl_id) { // RANDOM!

                    random_args = (bh_random_type*)instr->userfunc;
                    if (BH_SUCCESS != bh_data_malloc(random_args->operand[0])) {
                        std::cout << "SHIT HIT THE FAN" << std::endl;
                    }
                    sprintf(
                        symbol_c,
                        "BH_RANDOM_D_%s",
                        bhtype_to_shorthand(random_args->operand[0]->type)
                    );
                    symbol = std::string(symbol_c);

                    cres = target->symbol_ready(symbol);
                    if (!cres) {
                        sourcecode = "";

                        dict.SetValue("SYMBOL",     symbol);
                        dict.SetValue("TYPE_A0",    bhtype_to_ctype(random_args->operand[0]->type));
                        dict.SetValue("TYPE_A0_SHORTHAND", bhtype_to_shorthand(random_args->operand[0]->type));
                        sprintf(snippet_fn, "%s/random.tpl", snippet_path);
                        //sprintf(snippet_fn, "%s/random.omp.tpl", snippet_path);
                        ctemplate::ExpandTemplate(
                            snippet_fn,
                            ctemplate::STRIP_BLANK_LINES, 
                            &dict, 
                            &sourcecode
                        );
                        cres = target->compile(symbol, sourcecode.c_str(), sourcecode.size());
                    }

                    if (!cres) {
                        res = BH_ERROR;
                    } else {
                        // De-assemble the RANDOM_UFUNC     // CALL
                        target->funcs[symbol](0,
                            bh_base_array(random_args->operand[0])->data,
                            bh_nelements(random_args->operand[0]->ndim, random_args->operand[0]->shape)
                        );
                        res = BH_SUCCESS;
                    }

                } else if(instr->userfunc->id == matmul_impl_id) {
                    res = matmul_impl(instr->userfunc, NULL);
                } else if(instr->userfunc->id == nselect_impl_id) {
                    res = nselect_impl(instr->userfunc, NULL);
                } else {                            // Unsupported userfunc
                    res = BH_USERFUNC_NOT_SUPPORTED;
                }

                break;

            // Partial Reductions
            case BH_ADD_REDUCE:
            case BH_MULTIPLY_REDUCE:
            case BH_MINIMUM_REDUCE:
            case BH_MAXIMUM_REDUCE:
            case BH_LOGICAL_AND_REDUCE:
            case BH_BITWISE_AND_REDUCE:
            case BH_LOGICAL_OR_REDUCE:
            case BH_BITWISE_OR_REDUCE:

                sprintf(symbol_c, "%s_DD_%s%s",
                    bh_opcode_text(instr->opcode),
                    bhtype_to_shorthand(instr->operand[0]->type),
                    bhtype_to_shorthand(instr->operand[1]->type)
                );
                symbol = std::string(symbol_c);

                cres = target->symbol_ready(symbol);
                if (!cres) {
                    sourcecode = "";

                    dict.SetValue("OPERATOR", bhopcode_to_cexpr(instr->opcode));
                    dict.SetValue("SYMBOL", symbol);
                    dict.SetValue("TYPE_A0", bhtype_to_ctype(instr->operand[0]->type));
                    dict.SetValue("TYPE_A1", bhtype_to_ctype(instr->operand[1]->type));

                    sprintf(snippet_fn, "%s/reduction.tpl", snippet_path);
                    //sprintf(snippet_fn, "%s/reduction.omp.tpl", snippet_path);
                    ctemplate::ExpandTemplate(
                        snippet_fn,
                        ctemplate::STRIP_BLANK_LINES,
                        &dict,
                        &sourcecode
                    );
                    cres = target->compile(symbol, sourcecode.c_str(), sourcecode.size());
                }

                if (!cres) {
                    res = BH_ERROR;
                } else {    // CALL
                    target->funcs[symbol](0,
                        bh_base_array(instr->operand[0])->data,
                        instr->operand[0]->start,
                        instr->operand[0]->stride,
                        instr->operand[0]->shape,
                        instr->operand[0]->ndim,

                        bh_base_array(instr->operand[1])->data,
                        instr->operand[1]->start,
                        instr->operand[1]->stride,
                        instr->operand[1]->shape,
                        instr->operand[1]->ndim,

                        instr->constant.value
                    );
                    res = BH_SUCCESS;
                }

                break;

            // Binary elementwise: ADD, MULTIPLY...
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

                dims = instr->operand[0]->ndim;

                if (bh_is_constant(instr->operand[2])) {
                    sprintf(symbol_c, "%s_%ldD_DDC_%s%s%s",
                        bh_opcode_text(instr->opcode),
                        dims,
                        bhtype_to_shorthand(instr->operand[0]->type),
                        bhtype_to_shorthand(instr->operand[1]->type),
                        bhtype_to_shorthand(instr->constant.type)
                    );
                } else if(bh_is_constant(instr->operand[1])) {
                    sprintf(symbol_c, "%s_%ldD_DCD_%s%s%s",
                        bh_opcode_text(instr->opcode),
                        dims,
                        bhtype_to_shorthand(instr->operand[0]->type),
                        bhtype_to_shorthand(instr->constant.type),
                        bhtype_to_shorthand(instr->operand[2]->type)
                    );
                } else {
                    sprintf(symbol_c, "%s_%ldD_DDD_%s%s%s",
                        bh_opcode_text(instr->opcode),
                        dims,
                        bhtype_to_shorthand(instr->operand[0]->type),
                        bhtype_to_shorthand(instr->operand[1]->type),
                        bhtype_to_shorthand(instr->operand[2]->type)
                    );
                }
                symbol = std::string(symbol_c);
                
                cres = target->symbol_ready(symbol);
                if (!cres) {
                    sourcecode = "";
                    dict.SetValue("OPERATOR", bhopcode_to_cexpr(instr->opcode));
                    dict.ShowSection("binary");
                    if (bh_is_constant(instr->operand[2])) {
                        dict.SetValue("SYMBOL", symbol);
                        dict.SetValue("STRUCT_IN1", "D");
                        dict.SetValue("STRUCT_IN2", "C");
                        dict.SetValue("TYPE_OUT", bhtype_to_ctype(instr->operand[0]->type));
                        dict.SetValue("TYPE_IN1", bhtype_to_ctype(instr->operand[1]->type));
                        dict.SetValue("TYPE_IN2", bhtype_to_ctype(instr->constant.type));
                        dict.ShowSection("a1_dense");
                        dict.ShowSection("a2_scalar");
                    } else if (bh_is_constant(instr->operand[1])) {
                        dict.SetValue("SYMBOL", symbol);
                        dict.SetValue("STRUCT_IN1", "C");
                        dict.SetValue("STRUCT_IN2", "D");
                        dict.SetValue("TYPE_OUT", bhtype_to_ctype(instr->operand[0]->type));
                        dict.SetValue("TYPE_IN1", bhtype_to_ctype(instr->constant.type));
                        dict.SetValue("TYPE_IN2", bhtype_to_ctype(instr->operand[2]->type));
                        dict.ShowSection("a1_scalar");
                        dict.ShowSection("a2_dense");
                    } else {
                        dict.SetValue("SYMBOL", symbol);
                        dict.SetValue("STRUCT_IN1", "D");
                        dict.SetValue("STRUCT_IN2", "D");
                        dict.SetValue("TYPE_OUT", bhtype_to_ctype(instr->operand[0]->type));
                        dict.SetValue("TYPE_IN1", bhtype_to_ctype(instr->operand[1]->type));
                        dict.SetValue("TYPE_IN2", bhtype_to_ctype(instr->operand[2]->type));
                        dict.ShowSection("a1_dense");
                        dict.ShowSection("a2_dense");
                    }
                    if (1 == dims) {
                        sprintf(snippet_fn, "%s/traverse.1d.tpl", snippet_path);
                    } else if (2 == dims) {
                        sprintf(snippet_fn, "%s/traverse.2d.tpl", snippet_path);
                    } else if (3 == dims) {
                        sprintf(snippet_fn, "%s/traverse.3d.tpl", snippet_path);
                    } else {
                        sprintf(snippet_fn, "%s/traverse.naive.tpl", snippet_path);
                    }
                    //sprintf(snippet_fn, "%s/traverse.omp.tpl", snippet_path);
                    ctemplate::ExpandTemplate(
                        snippet_fn,
                        ctemplate::STRIP_BLANK_LINES,
                        &dict,
                        &sourcecode
                    );
                }

                cres = target->compile(symbol, sourcecode.c_str(), sourcecode.size());
                if (cres) { // CALL
                    if (bh_is_constant(instr->operand[2])) {         // DDC
                        target->funcs[symbol](0,
                            bh_base_array(instr->operand[0])->data,
                            instr->operand[0]->start,
                            instr->operand[0]->stride,

                            bh_base_array(instr->operand[1])->data,
                            instr->operand[1]->start,
                            instr->operand[1]->stride,

                            &(instr->constant.value),

                            instr->operand[0]->shape,
                            instr->operand[0]->ndim
                        );
                    } else if (bh_is_constant(instr->operand[1])) {  // DCD
                        target->funcs[symbol](0,
                            bh_base_array(instr->operand[0])->data,
                            instr->operand[0]->start,
                            instr->operand[0]->stride,

                            &(instr->constant.value),

                            bh_base_array(instr->operand[2])->data,
                            instr->operand[2]->start,
                            instr->operand[2]->stride,

                            instr->operand[0]->shape,
                            instr->operand[0]->ndim
                        );
                    } else {                                        // DDD
                        target->funcs[symbol](0,
                            bh_base_array(instr->operand[0])->data,
                            instr->operand[0]->start,
                            instr->operand[0]->stride,

                            bh_base_array(instr->operand[1])->data,
                            instr->operand[1]->start,
                            instr->operand[1]->stride,

                            bh_base_array(instr->operand[2])->data,
                            instr->operand[2]->start,
                            instr->operand[2]->stride,

                            instr->operand[0]->shape,
                            instr->operand[0]->ndim
                        );
                    }
                    
                    res = BH_SUCCESS;
                } else {
                    res = BH_ERROR;
                }

                break;

            // Unary elementwise: SQRT, SIN...
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

                if (bh_is_constant(instr->operand[1])) {
                    sprintf(symbol_c, "%s_%ldD_DC_%s%s",
                            bh_opcode_text(instr->opcode),
                            dims,
                            bhtype_to_shorthand(instr->operand[0]->type),
                            bhtype_to_shorthand(instr->constant.type)
                    );
                } else {
                    sprintf(symbol_c, "%s_%ldD_DD_%s%s",
                            bh_opcode_text(instr->opcode),
                            dims,
                            bhtype_to_shorthand(instr->operand[0]->type),
                            bhtype_to_shorthand(instr->operand[1]->type)
                    );
                }
                symbol = std::string(symbol_c);

                cres = target->symbol_ready(symbol);
                if (!cres) {    // SNIPPET
                    sourcecode = "";
                    dict.SetValue("OPERATOR", bhopcode_to_cexpr(instr->opcode));
                    dict.ShowSection("unary");
                    if (bh_is_constant(instr->operand[1])) {
                        dict.SetValue("SYMBOL", symbol);
                        dict.SetValue("STRUCT_IN1", "C");
                        dict.SetValue("TYPE_OUT", bhtype_to_ctype(instr->operand[0]->type));
                        dict.SetValue("TYPE_IN1", bhtype_to_ctype(instr->constant.type));
                        dict.ShowSection("a1_scalar");
                    } else {
                        dict.SetValue("SYMBOL", symbol);
                        dict.SetValue("STRUCT_IN1", "D");
                        dict.SetValue("TYPE_OUT", bhtype_to_ctype(instr->operand[0]->type));
                        dict.SetValue("TYPE_IN1", bhtype_to_ctype(instr->operand[1]->type));
                        dict.ShowSection("a1_dense");
                    } 
                    if (1 == dims) {
                        sprintf(snippet_fn, "%s/traverse.1d.tpl", snippet_path);
                    } else if (2 == dims) {
                        sprintf(snippet_fn, "%s/traverse.2d.tpl", snippet_path);
                    } else if (3 == dims) {
                        sprintf(snippet_fn, "%s/traverse.3d.tpl", snippet_path);
                    } else {
                        sprintf(snippet_fn, "%s/traverse.naive.tpl", snippet_path);
                    }
                    //sprintf(snippet_fn, "%s/traverse.omp.tpl", snippet_path);
                    ctemplate::ExpandTemplate(
                        snippet_fn,
                        ctemplate::STRIP_BLANK_LINES,
                        &dict,
                        &sourcecode
                    );
                }

                cres = target->compile(symbol, sourcecode.c_str(), sourcecode.size());
                if (!cres) {
                    res = BH_ERROR;
                } else {    // CALL
                    if (bh_is_constant(instr->operand[1])) {
                        target->funcs[symbol](0,
                            bh_base_array(instr->operand[0])->data,
                            instr->operand[0]->start,
                            instr->operand[0]->stride,

                            &(instr->constant.value),

                            instr->operand[0]->shape,
                            instr->operand[0]->ndim
                        );
                    } else {
                        target->funcs[symbol](0,
                            bh_base_array(instr->operand[0])->data,
                            instr->operand[0]->start,
                            instr->operand[0]->stride,

                            bh_base_array(instr->operand[1])->data,
                            instr->operand[1]->start,
                            instr->operand[1]->stride,

                            instr->operand[0]->shape,
                            instr->operand[0]->ndim
                        );
                    }
                    res = BH_SUCCESS;
                }
                break;

            default:                            // Shit hit the fan
                res = bh_compute_apply_naive(instr);

        }

        if (BH_SUCCESS != res) {    // Instruction failed
            break;
        }
    }

	return res;
}

bh_error bh_ve_dynamite_shutdown(void)
{
    bh_vcache_clear();  // De-allocate the malloc-cache
    bh_vcache_delete();

    delete target;

    return BH_SUCCESS;
}

bh_error bh_ve_dynamite_reg_func(char *fun, bh_intp *id) 
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

bh_error bh_matmul( bh_userfunc *arg, void* ve_arg)
{
    return bh_compute_matmul( arg, ve_arg );
}

bh_error bh_nselect( bh_userfunc *arg, void* ve_arg)
{
    return bh_compute_nselect( arg, ve_arg );
}

