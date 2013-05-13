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
#include "targets.cpp"
#include <string>

static bh_component *myself = NULL;
static bh_userfunc_impl random_impl = NULL;
static bh_intp random_impl_id = 0;
static bh_userfunc_impl matmul_impl = NULL;
static bh_intp matmul_impl_id = 0;
static bh_userfunc_impl nselect_impl = NULL;
static bh_intp nselect_impl_id = 0;

static bh_intp vcache_size   = 10;

char* target_cmd;   // Dynamite Arguments
char* kernel_path;
char* object_path;
char* snippet_path;

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
    target_cmd = getenv("BH_VE_DYNAMITE_TARGET");           // For the compiler
    if (NULL==target_cmd) {
        assign_string(target_cmd, "gcc -O2 -march=native -fPIC -x c -shared - -o ");
    }

    object_path = getenv("BH_VE_DYNAMITE_OBJECT_PATH");
    if (NULL==object_path) {
        assign_string(object_path, "objects/object_XXXXXX");
    }

    kernel_path = getenv("BH_VE_DYNAMITE_KERNEL_PATH");
    if (NULL==kernel_path) {
        assign_string(kernel_path, "kernels/kernel_XXXXXX");
    }

    snippet_path = getenv("BH_VE_DYNAMITE_SNIPPET_PATH");   // For the sorucecode-generator
    if (NULL==snippet_path) {
        assign_string(snippet_path, "snippets/");
    }

    return BH_SUCCESS;
}

bh_error bh_ve_dynamite_execute(bh_intp instruction_count, bh_instruction* instruction_list)
{
    bh_intp count;
    bh_instruction* inst;
    bh_error res = BH_SUCCESS;

    process target(target_cmd, object_path, kernel_path);

    for (count=0; count<instruction_count; count++) {

        ctemplate::TemplateDictionary dict("example");
        std::string sourcecode;

        char symbol[200],
             type_out[30],
             type_in1[30],
             type_in2[30],
             operator_src[20];
        char *opcode_txt;
        bool cres;

        inst = &instruction_list[count];

        res = bh_vcache_malloc(inst);          // Allocate memory for operands
        if (BH_SUCCESS != res) {
            printf("Unhandled error returned by bh_vcache_malloc() called from bh_ve_dynamite_execute()\n");
            return res;
        }
                                                    
        switch (inst->opcode) {                     // Dispatch instruction

            case BH_NONE:                        // NOOP.
            case BH_DISCARD:
            case BH_SYNC:
                res = BH_SUCCESS;
                break;
            case BH_FREE:                        // Store data-pointer in malloc-cache
                res = bh_vcache_free( inst );
                break;

            case BH_ADD:
            case BH_SUBTRACT:
            case BH_MULTIPLY:
            case BH_DIVIDE:
            case BH_MOD:
            case BH_BITWISE_AND:
            case BH_BITWISE_OR:
            case BH_BITWISE_XOR:
            case BH_LEFT_SHIFT:
            case BH_RIGHT_SHIFT:
            case BH_EQUAL:
            case BH_NOT_EQUAL:
            case BH_GREATER:
            case BH_GREATER_EQUAL:
            case BH_LESS:
            case BH_LESS_EQUAL:
            case BH_LOGICAL_AND:
            case BH_LOGICAL_OR:

                assign_string(opcode_txt, bh_opcode_text(inst->opcode));
                sourcecode = "";
                strcpy(operator_src, opcode_to_opstr(inst->opcode));

                dict.SetValue("OPERATOR",       operator_src);
                dict.SetValue("OPCODE_NAME",    opcode_txt);

                if (bh_is_constant(inst->operand[2])) {
                    strcpy(type_out, type_text(inst->operand[0]->type));
                    strcpy(type_in1, type_text(inst->operand[1]->type));
                    strcpy(type_in2, type_text(inst->constant.type));
                    sprintf(symbol, "%s_D%s%s_%s%s%s", opcode_txt, "D", "C", 
                            type_out, 
                            type_in1, 
                            type_in2
                    );
                    dict.SetValue("STRUCT_IN1", "D");
                    dict.SetValue("STRUCT_IN2", "C");
                    dict.SetValue("TYPE_OUT", type_out);
                    dict.SetValue("TYPE_IN1", type_in1);
                    dict.SetValue("TYPE_IN2", type_in2);
                    dict.ShowSection("a1_dense");
                    dict.ShowSection("a2_scalar");
                } else if (bh_is_constant(inst->operand[1])) {
                    strcpy(type_out, type_text(inst->operand[0]->type));
                    strcpy(type_in1, type_text(inst->constant.type));
                    strcpy(type_in2, type_text(inst->operand[2]->type));
                    sprintf(symbol, "%s_D%s%s_%s%s%s", opcode_txt, "C", "D",
                        type_out,
                        type_in1,
                        type_in2
                    );
                    dict.SetValue("STRUCT_IN1", "C");
                    dict.SetValue("STRUCT_IN2", "D");
                    dict.SetValue("TYPE_OUT", type_out);
                    dict.SetValue("TYPE_IN1", type_in1);
                    dict.SetValue("TYPE_IN2", type_in2);
                    dict.ShowSection("a1_scalar");
                    dict.ShowSection("a2_dense");
                } else {
                    strcpy(type_out, type_text(inst->operand[0]->type));
                    strcpy(type_in1, type_text(inst->operand[1]->type));
                    strcpy(type_in2, type_text(inst->operand[2]->type));
                    sprintf(symbol, "%s_D%s%s_%s%s%s", opcode_txt, "D", "D",
                        type_text(inst->operand[0]->type),
                        type_text(inst->operand[1]->type),
                        type_text(inst->operand[2]->type)
                    );
                    dict.SetValue("STRUCT_IN1", "D");
                    dict.SetValue("STRUCT_IN2", "D");
                    dict.SetValue("TYPE_OUT", type_out);
                    dict.SetValue("TYPE_IN1", type_in1);
                    dict.SetValue("TYPE_IN2", type_in2);
                    dict.ShowSection("a1_dense");
                    dict.ShowSection("a2_dense");
                }
            
                ctemplate::ExpandTemplate("snippets/traverse_DDD.tpl", ctemplate::DO_NOT_STRIP, &dict, &sourcecode);
                cres = target.compile(symbol, sourcecode.c_str(), sourcecode.size());

                if (cres) {
                    if (bh_is_constant(inst->operand[2])) {         // DDC
                        target.f(0,
                            inst->operand[0]->start, inst->operand[0]->stride,
                            inst->operand[0]->data,
                            inst->operand[1]->start, inst->operand[1]->stride,
                            inst->operand[1]->data,
                            &(inst->constant.value),
                            inst->operand[0]->shape, inst->operand[0]->ndim,
                            bh_nelements(inst->operand[0]->ndim, inst->operand[0]->shape)
                        );
                    } else if (bh_is_constant(inst->operand[1])) {  // DCD
                        target.f(0,
                            inst->operand[0]->start, inst->operand[0]->stride,
                            inst->operand[0]->data,
                            &(inst->constant.value),
                            inst->operand[2]->start, inst->operand[2]->stride,
                            inst->operand[2]->data,
                            inst->operand[0]->shape, inst->operand[0]->ndim,
                            bh_nelements(inst->operand[0]->ndim, inst->operand[0]->shape)
                        );
                    } else {                                        // DDD
                        target.f(0,
                            inst->operand[0]->start, inst->operand[0]->stride,
                            inst->operand[0]->data,
                            inst->operand[1]->start, inst->operand[1]->stride,
                            inst->operand[1]->data,
                            inst->operand[2]->start, inst->operand[2]->stride,
                            inst->operand[2]->data,
                            inst->operand[0]->shape, inst->operand[0]->ndim,
                            bh_nelements(inst->operand[0]->ndim, inst->operand[0]->shape)
                        );
                    }
                    
                    res = BH_SUCCESS;
                } else {
                    res = BH_ERROR;
                }

                break;

            case BH_USERFUNC:                    // External libraries

                if(inst->userfunc->id == random_impl_id) {
                    res = random_impl(inst->userfunc, NULL);
                } else if(inst->userfunc->id == matmul_impl_id) {
                    res = matmul_impl(inst->userfunc, NULL);
                } else if(inst->userfunc->id == nselect_impl_id) {
                    res = nselect_impl(inst->userfunc, NULL);
                } else {                            // Unsupported userfunc
                    res = BH_USERFUNC_NOT_SUPPORTED;
                }

                break;

            default:                            // Built-in operations
                res = bh_compute_apply_naive( inst );

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

    return BH_SUCCESS;
}

bh_error bh_ve_dynamite_reg_func(char *fun, bh_intp *id) 
{
    if(strcmp("bh_random", fun) == 0)
    {
    	if (random_impl == NULL)
    	{
			bh_component_get_func(myself, fun, &random_impl);
			if (random_impl == NULL)
				return BH_USERFUNC_NOT_SUPPORTED;

			random_impl_id = *id;
			return BH_SUCCESS;			
        }
        else
        {
        	*id = random_impl_id;
        	return BH_SUCCESS;
        }
    }
    else if(strcmp("bh_matmul", fun) == 0)
    {
    	if (matmul_impl == NULL)
    	{
            bh_component_get_func(myself, fun, &matmul_impl);
            if (matmul_impl == NULL)
                return BH_USERFUNC_NOT_SUPPORTED;
            
            matmul_impl_id = *id;
            return BH_SUCCESS;			
        }
        else
        {
        	*id = matmul_impl_id;
        	return BH_SUCCESS;
        }
    }
    else if(strcmp("bh_nselect", fun) == 0)
    {
        if (nselect_impl == NULL)
        {
            bh_component_get_func(myself, fun, &nselect_impl);
            if (nselect_impl == NULL)
                return BH_USERFUNC_NOT_SUPPORTED;
            
            nselect_impl_id = *id;
            return BH_SUCCESS;
        }
        else
        {
            *id = nselect_impl_id;
            return BH_SUCCESS;
        }
    }
        
    return BH_USERFUNC_NOT_SUPPORTED;
}

bh_error bh_random( bh_userfunc *arg, void* ve_arg)
{
    return bh_compute_random( arg, ve_arg );
}

bh_error bh_matmul( bh_userfunc *arg, void* ve_arg)
{
    return bh_compute_matmul( arg, ve_arg );
}

bh_error bh_nselect( bh_userfunc *arg, void* ve_arg)
{
    return bh_compute_nselect( arg, ve_arg );
}
