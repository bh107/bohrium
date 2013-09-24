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
#include "bh_ve_tiling.h"
#include <iostream>
#include <bh_vcache.h>

static bh_component *myself = NULL;
static bh_userfunc_impl random_impl = NULL;
static bh_intp random_impl_id = 0;
static bh_userfunc_impl matmul_impl = NULL;
static bh_intp matmul_impl_id = 0;
static bh_userfunc_impl lu_impl = NULL;
static bh_intp lu_impl_id = 0;
static bh_userfunc_impl fft_impl = NULL;
static bh_intp fft_impl_id = 0;
static bh_userfunc_impl fft2_impl = NULL;
static bh_intp fft2_impl_id = 0;

static bh_intp bh_ve_tiling_buffersizes = 0;
static bh_computeloop_naive* bh_ve_tiling_compute_loops = NULL;
static bh_tstate_naive* bh_ve_tiling_tstates = NULL;

static bh_intp block_size    = 7000;
static bh_intp bin_max       = 20;
static bh_intp base_max      = 7;
static bh_intp vcache_size   = 10;

bh_error bh_ve_tiling_init(bh_component *self)
{
    myself = self;                              // Assign config container.

    char *env = getenv("BH_VE_TILING_BLOCKSIZE");   // Override block_size from environment-variable.
    if (env != NULL){
        block_size = atoi(env);
    }
    if (block_size <= 0) {                        // Verify it
        fprintf(stderr, "BH_VE_TILING_BLOCKSIZE (%ld) should be greater than zero!\n", (long int)block_size);
        return BH_ERROR;
    }

    env = getenv("BH_VE_TILING_BINMAX");      // Override block_size from environment-variable.
    if (env != NULL) {
        bin_max = atoi(env);
    }
    if (bin_max <= 0) {                            // Verify it
        fprintf(stderr, "BH_VE_TILING_BINMAX (%ld) should be greater than zero!\n", (long int)bin_max);
        return BH_ERROR;
    }

    env = getenv("BH_VE_TILING_BASEMAX");        // Override base max from environment-variable.
    if(env != NULL) {
        base_max = atoi(env);
    }
    if (base_max <= 0) {                        // Verify it
        fprintf(stderr, "BH_VE_TILING_BASEMAX (%ld) should be greater than zero!\n", (long int)base_max);
        return BH_ERROR;
    }

    env = getenv("BH_CORE_VCACHE_SIZE");    // Override block_size from environment-variable.
    if(env != NULL) {
        vcache_size = atoi(env);
    }
    if (vcache_size < 0) {                  // Verify it
        fprintf(stderr, "BH_CORE_VCACHE_SIZE (%ld) should be greater than zero!\n", (long int)vcache_size);
        return BH_ERROR;
    }

    bh_vcache_init(vcache_size);
    return BH_SUCCESS;
}

inline bh_error block_execute(bh_instruction* instr, bh_intp start, bh_intp end) {

    bh_intp i, k;

    if ((end - start + 1) > bh_ve_tiling_buffersizes) { // Make sure we have enough space
    	bh_intp mcount = (end - start + 1);

    	if (mcount % 1000 != 0) {                       // We only work in multiples of 1000
    		mcount = (mcount + 1000) - (mcount % 1000);
        }

    	bh_ve_tiling_buffersizes = 0;                   // Make sure we re-allocate on error

    	if (bh_ve_tiling_compute_loops != NULL) {
    		free(bh_ve_tiling_compute_loops);
    		free(bh_ve_tiling_tstates);
    		bh_ve_tiling_compute_loops = NULL;
    		bh_ve_tiling_tstates = NULL;
    	}

    	bh_ve_tiling_compute_loops    = (bh_computeloop_naive*)malloc(sizeof(bh_computeloop_naive) * mcount);
        bh_ve_tiling_tstates          = (bh_tstate_naive*)malloc(sizeof(bh_tstate_naive)*mcount);

    	if (bh_ve_tiling_compute_loops == NULL) {
    		return BH_OUT_OF_MEMORY;
        }
    	if (bh_ve_tiling_tstates == NULL) {
    		return BH_OUT_OF_MEMORY;
        }

    	bh_ve_tiling_buffersizes = mcount;
    }

    bh_computeloop_naive* compute_loops = bh_ve_tiling_compute_loops;
    bh_tstate_naive* states = bh_ve_tiling_tstates;
    bh_intp  nelements, trav_end=0;
    bh_error ret_errcode = BH_SUCCESS;

    for(i=0; i<= (end-start);i++) {                 // Reset traversal coordinates
        bh_tstate_reset_naive(&states[i]);
    }

    for(i=start, k=0; i <= end; i++,k++) {          // Get the compute-loops
        switch(instr[i].opcode) {
            case BH_DISCARD:                        // Ignore sys-ops
            case BH_FREE:
            case BH_SYNC:
            case BH_NONE:
            case BH_USERFUNC:
                break;

            default:
                compute_loops[k] = bh_compute_get_naive(&instr[i]);
                if(compute_loops[k] == NULL) {
                    return BH_TYPE_NOT_SUPPORTED;
                }
        }
    }
                                                    // Block-size split
    nelements = bh_nelements(
        instr[start].operand[0].ndim,
        instr[start].operand[0].shape
    );
    while (nelements>trav_end) {
        trav_end    += block_size;
        if (trav_end > nelements) {
            trav_end = nelements;
        }

        for(i=start, k=0; i <= end; i++, k++) {
            switch(instr[i].opcode) {
                case BH_DISCARD:                    // Ignore sys-ops
                case BH_FREE:
                case BH_SYNC:
                case BH_NONE:
                case BH_USERFUNC:
                    break;

                default:
                    compute_loops[k](&instr[i], &states[k], trav_end);
            }
        }
    }

    return ret_errcode;
}

bh_error bh_ve_tiling_execute(bh_ir* bhir)
{
    bh_intp cur_index,  j;
    bh_instruction *inst, *binst;

    bh_intp bin_start, bin_end, bin_size;
    bh_intp bundle_start, bundle_end, bundle_size;
    bh_error res = BH_SUCCESS;

    bh_intp instruction_count = bhir->ninstr;
    bh_instruction* instruction_list = bhir->instr_list;

    for(cur_index=0; cur_index<instruction_count; cur_index++) {
        inst = &instruction_list[cur_index];

        res = bh_vcache_malloc(inst);       // Allocate memory for operands
        if (res != BH_SUCCESS) {
            return res;
        }
        switch(inst->opcode) {              // Dispatch instruction

            case BH_NONE:                   // NOOP.
            case BH_DISCARD:
            case BH_SYNC:
                res = BH_SUCCESS;
                break;

            case BH_FREE:                   // Store data-pointer in malloc-cache
                res = bh_vcache_free(inst);
                break;

            case BH_ADD_REDUCE:
            case BH_MULTIPLY_REDUCE:
            case BH_MINIMUM_REDUCE:
            case BH_MAXIMUM_REDUCE:
            case BH_LOGICAL_AND_REDUCE:
            case BH_LOGICAL_OR_REDUCE:
            case BH_LOGICAL_XOR_REDUCE:
            case BH_BITWISE_AND_REDUCE:
            case BH_BITWISE_OR_REDUCE:
            case BH_BITWISE_XOR_REDUCE:
				res = bh_compute_reduce_naive(inst);
            	break;

            case BH_USERFUNC:               // External libraries

                if (inst->userfunc->id == random_impl_id) {
                    res = random_impl(inst->userfunc, NULL);
                } else if(inst->userfunc->id == matmul_impl_id) {
                    res = matmul_impl(inst->userfunc, NULL);
                } else if(inst->userfunc->id == lu_impl_id) {
                    res = lu_impl(inst->userfunc, NULL);
                } else if(inst->userfunc->id == fft_impl_id) {
                    res = fft_impl(inst->userfunc, NULL);
                } else if(inst->userfunc->id == fft2_impl_id) {
                    res = fft2_impl(inst->userfunc, NULL);
                } else {
                    res = BH_USERFUNC_NOT_SUPPORTED;    // Unsupported userfunc
                }

                break;

            default:                            // Built-in operations

                //
                // -={[ BINNING ]}=-
                // Count built-ins and their start/end indexes and
                // allocate memory for them.
                //
                bin_start   = cur_index;
                bin_end     = cur_index;
                bool has_sys = false;
                for(j=bin_start+1; (j<instruction_count) && (j<bin_start+1+bin_max); j++) {
                    binst = &instruction_list[j];

                    if ((binst->opcode == BH_USERFUNC)              || \
                        (binst->opcode == BH_ADD_REDUCE)            || \
                        (binst->opcode == BH_MULTIPLY_REDUCE)       || \
                        (binst->opcode == BH_MINIMUM_REDUCE)        || \
                        (binst->opcode == BH_MAXIMUM_REDUCE)        || \
                        (binst->opcode == BH_LOGICAL_AND_REDUCE)    || \
                        (binst->opcode == BH_LOGICAL_OR_REDUCE)     || \
                        (binst->opcode == BH_LOGICAL_XOR_REDUCE)    || \
                        (binst->opcode == BH_BITWISE_AND_REDUCE)    || \
                        (binst->opcode == BH_BITWISE_OR_REDUCE)     || \
                        (binst->opcode == BH_BITWISE_XOR_REDUCE)) {
                        break;          // Cannot bin reductions and userfuncs.
                    }
                                                                // Delay sys-op
                    if ((binst->opcode == BH_NONE) || \
                        (binst->opcode == BH_DISCARD) || \
                        (binst->opcode == BH_SYNC) || \
                        (binst->opcode == BH_FREE)
                        ) {
                        has_sys = true;
                    }

                    bin_end++;                                  // The "end" index
                    res = bh_vcache_malloc(binst);              // Allocate memory for operands
                    if (res != BH_SUCCESS) {
                        return res;
                    }
                }
                bin_size = bin_end - bin_start +1;              // The counting part

                //
                // -={[ BUNDLING ]}=-
                //
                // Determine whether the list of instructions can be executed in a tiled fashion.
                //
                bundle_size     = (bin_size > 1) ? bh_inst_bundle(instruction_list, bin_start, bin_end, base_max) : 1;
                bundle_start    = bin_start;
                bundle_end      = bundle_start + bundle_size-1;

                //printf("Bundle: %ld %ld -> %ld\n", bundle_size, bundle_start, bundle_end);
                if (bundle_size > 1) {                  // Execute in a tiled fashion
                    block_execute(instruction_list, bundle_start, bundle_end);

                    if (has_sys) {                      // De-allocate memory for operands.
                        for(j=bundle_start; j <= bundle_end; j++) {
                            inst = &instruction_list[j];
                            switch(inst->opcode) {
                                case BH_NONE:           // NOOP / ignore.
                                case BH_DISCARD:
                                case BH_SYNC:
                                    res = BH_SUCCESS;
                                    break;

                                case BH_FREE:           // De-allocate
                                    res = bh_vcache_free(inst);
                                    break;
                            }

                            if (res != BH_SUCCESS) {    // De-allocation failed.
                            	break;
                            }
                        }
                    }

                    cur_index += bundle_size-1;
                } else {                                // Regular execution
                    inst = &instruction_list[bundle_start];
                    res  = bh_compute_apply_naive(inst);
                }
                /*
                fprintf(stderr,
                        "Hmm %ld bin_size=%ld, bundle_size=%ld\n",
                        cur_index, bin_size, bundle_size
                );*/
        }

        if (res != BH_SUCCESS) {    // Instruction failed
            break;
        }
    }

	return res;
}

bh_error bh_ve_tiling_shutdown(void)
{
    if (vcache_size>0) {
        bh_vcache_clear();
        bh_vcache_delete();
    }
    return BH_SUCCESS;
}

bh_error bh_ve_tiling_reg_func(char *fun, bh_intp *id)
{
    if (strcmp("bh_random", fun) == 0) {
    	if (random_impl == NULL) {
			bh_component_get_func(myself, fun, &random_impl);
			if (random_impl == NULL) {
				return BH_USERFUNC_NOT_SUPPORTED;
            }

			random_impl_id = *id;
			return BH_SUCCESS;
        } else {
        	*id = random_impl_id;
        	return BH_SUCCESS;
        }
    } else if(strcmp("bh_matmul", fun) == 0) {
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
    } else if(strcmp("bh_lu", fun) == 0) {
    	if (lu_impl == NULL) {
			bh_component_get_func(myself, fun, &lu_impl);
			if (lu_impl == NULL) {
				return BH_USERFUNC_NOT_SUPPORTED;
            }

			lu_impl_id = *id;
			return BH_SUCCESS;
        } else {
        	*id = lu_impl_id;
        	return BH_SUCCESS;
        }
    } else if(strcmp("bh_fft", fun) == 0) {
    	if (fft_impl == NULL) {
			bh_component_get_func(myself, fun, &fft_impl);
			if (fft_impl == NULL) {
				return BH_USERFUNC_NOT_SUPPORTED;
            }

			fft_impl_id = *id;
			return BH_SUCCESS;
        } else {
        	*id = fft_impl_id;
        	return BH_SUCCESS;
        }
    } else if(strcmp("bh_fft2", fun) == 0) {
    	if (fft2_impl == NULL) {
			bh_component_get_func(myself, fun, &fft2_impl);
			if (fft2_impl == NULL) {
				return BH_USERFUNC_NOT_SUPPORTED;
            }

			fft2_impl_id = *id;
			return BH_SUCCESS;
        } else {
        	*id = fft2_impl_id;
        	return BH_SUCCESS;
        }
    }

    return BH_USERFUNC_NOT_SUPPORTED;
}

bh_error bh_random( bh_userfunc *arg, void* ve_arg)
{
    return bh_compute_random(arg, ve_arg);
}

bh_error bh_matmul( bh_userfunc *arg, void* ve_arg)
{
    return bh_compute_matmul(arg, ve_arg);
}

