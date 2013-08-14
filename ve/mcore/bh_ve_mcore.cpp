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
#include "bh_ve_mcore.h"
#include <bh_vcache.h>
#include <iostream>

#include <queue>                                        //
#include <pthread.h>                                    // MULTICORE libraries
#include <semaphore.h>                                  //

#if _POSIX_BARRIERS <= 0
#error This system does not support POSIX barriers
#else

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

//Forward definition of the function
bh_error bh_reduce_impl( bh_userfunc *arg, void* ve_arg );

static bh_intp vcache_size = 10;
static bh_intp block_size = 1000;
                                                        //
                                                        // MULTICORE datastructures and variables
                                                        //

typedef struct worker_data {                            // Thread identity and control
    int id;
    bh_computeloop_naive loop;
    bh_instruction *instr;
    bh_tstate_naive *state;
    bh_index nelements;
} worker_data_t;

static pthread_barrier_t   work_start;                  // Work synchronization using barrier
static pthread_barrier_t   work_sync;                   // Work synchronization using barrier

static pthread_t        worker[MCORE_MAX_WORKERS];          // Worker-pool
static worker_data_t    worker_data[MCORE_MAX_WORKERS];     // And their associated data

static int worker_count = MCORE_WORKERS;
static bh_tstate_naive tstates[MCORE_MAX_WORKERS];

static void* job(void *worker_arg)
{
    int sync_res;
    worker_data_t *my_job = (worker_data_t*)worker_arg;     // Grab the thread argument

    DEBUG_PRINT("Worker %d - Started.\n", my_job->id);
    while(true) {
        sync_res = pthread_barrier_wait( &work_start );     // Wait for work
        if (!(sync_res == 0 || sync_res == PTHREAD_BARRIER_SERIAL_THREAD)) {
            DEBUG_PRINT("Worker %d - Error waiting for work! [ERRNO: %d]\n", my_job->id, sync_res);
        }
    
        if (my_job->instr != NULL) {                        // Got a job
            if (my_job->instr->opcode == BH_USERFUNC) {     // userfunc <--- wtf?
                bh_compute_apply_naive(my_job->instr);
            } else {                                        // built-in
                DEBUG_PRINT("Worker %d - Got a job...\n", my_job->id);
                (*my_job->loop)(my_job->instr, my_job->state, my_job->nelements);
                DEBUG_PRINT("Worker %d - Is done!\n", my_job->id);
                sync_res = pthread_barrier_wait( &work_sync );  // Wait for the others to finish
                if (!(sync_res == 0 || sync_res == PTHREAD_BARRIER_SERIAL_THREAD)) {
                    DEBUG_PRINT("Worker %d - Error synchronizing...\n", my_job->id);
                }
            }
        } else {                                            // EXIT!
            DEBUG_PRINT("Worker %d - An exit job...\n", my_job->id);
            break;
        }
    }
    DEBUG_PRINT("Worker %d - Exiting.\n", my_job->id);

    return NULL;

}

bh_error bh_ve_mcore_init(bh_component *self)
{
    char *env;
    int i;
    bh_error res = BH_SUCCESS;
    myself = self;                              // Assign config container.

    env = getenv("BH_CORE_VCACHE_SIZE");    // VCACHE size
    if (NULL != env) {
        vcache_size = atoi(env);
    }
    if (vcache_size < 0) {
        fprintf(stderr, "BH_CORE_VCACHE_SIZE (%ld) is invalid; "
                        "Use n>0 to set its size and n=0 to disable.\n",
                        (long int)vcache_size);
        return BH_ERROR;
    }
    bh_vcache_init(vcache_size);

    env = getenv("BH_VE_MCORE_BLOCKSIZE");         // Override block_size from ENVVAR
    if (env != NULL) {
        block_size = atoi(env);
    }
    if (block_size <= 0) {                      // Verify it
        fprintf(stderr, "BH_VE_MCORE_BLOCKSIZE (%ld) should be greater than zero!\n", (long)block_size);
        return BH_ERROR;
    }

    env = getenv("BH_VE_MCORE_NTHREADS");          // Override worker_count with ENVVAR
    if (env != NULL) {
        worker_count = atoi(env);
    }

    if (worker_count > MCORE_MAX_WORKERS) {     // Verify worker count
        fprintf(stderr,"BH_VE_MCORE_NTHREADS capped to %i.\n", MCORE_MAX_WORKERS);
        worker_count = MCORE_MAX_WORKERS;
    } else if (worker_count < 1) {
        fprintf(stderr,"BH_VE_MCORE_NTHREADS capped to default %i.\n", MCORE_WORKERS);
        worker_count = MCORE_WORKERS;
    }
                                                        //
                                                        // Multicore initialization
                                                        //

                                                        // Barriers for work syncronization
    if (pthread_barrier_init( &work_start, NULL, worker_count+1) != 0) {
        return BH_ERROR;
    }
    if (pthread_barrier_init( &work_sync, NULL, worker_count+1) != 0) {
        return BH_ERROR;
    }

    DEBUG_PRINT("[worker_count=%d, block_size=%lu]\n", worker_count, block_size);

    for(i=0; i<worker_count; i++) {                 // Worker-threads for job computation
#ifdef _WIN32
        worker_data[i].id = i;
        worker_data[i].loop = NULL;
        worker_data[i].instr = NULL;
        worker_data[i].state = NULL;
        worker_data[i].nelements = 0;
#else
        worker_data[i] = (worker_data_t){i, NULL, NULL, NULL, 0};
#endif
        if (pthread_create( &worker[i], NULL, job, &worker_data[i] ) != 0) {
            res = BH_ERROR;
            break;
        }
    }

    return res;

}

bh_error bh_ve_mcore_shutdown( void )
{
    int sync_res, i;

    DEBUG_PRINT("%s\n", "Sending shutdown signals to workers...");
    for(i=0; i<worker_count; i++) {
        worker_data[i].instr = NULL;
    }

    sync_res = pthread_barrier_wait( &work_start );
    if (!(sync_res == 0 || sync_res == PTHREAD_BARRIER_SERIAL_THREAD)) {
        DEBUG_PRINT("Error when opening work-start barrier [ERRNO: %d\n", sync_res);
    }

    DEBUG_PRINT("Waiting for workers...\n");
    for(i=0; i<worker_count; i++) {                 // Wait for termination
        DEBUG_PRINT("Waiting on %d...\n", i);
        sync_res = pthread_join( worker[i], NULL);
        if (sync_res != 0) {
            DEBUG_PRINT("ERROR joining thread. [ERRNO: %d].\n", sync_res);
        }
    }
    DEBUG_PRINT("Workers joined.\n");
                                                    // Cleanup syncronization primitives
    pthread_barrier_destroy(&work_start);
    pthread_barrier_destroy(&work_sync);

    // De-allocate the malloc-cache
    if (vcache_size>0) {
        bh_vcache_clear();
        bh_vcache_delete();
    }

    return BH_SUCCESS;
}

inline bh_error dispatch( bh_instruction* instr, bh_index nelements) {

    int sync_res;
    bh_computeloop_naive loop;
    bh_intp i;
    bh_index  last_dim, start, end, size;

    loop     = bh_compute_get_naive( instr );
    last_dim = instr->operand[0].ndim-1;
    size     = nelements / worker_count;

    for (i=0; i<worker_count;i++) {       // tstate = (0, 0, 0, ..., 0)
        bh_tstate_reset_naive(&tstates[i]);
    }
    while(tstates[worker_count-1].cur_e < nelements) {

        for(i=0;i<worker_count;i++) {               // WORDER SETUP
            start   = size * i;                     // Partition input
            end     = start + size;
            if (i == worker_count-1) {              // Last worker gets the remainder
                end += (nelements % worker_count);
            }

            tstates[i].cur_e = start;
            tstates[i].coord[last_dim] = start;

            worker_data[i].loop         = loop;     // Setup the remaining job-spec
            worker_data[i].instr        = instr;
            worker_data[i].state        = &tstates[i];
            worker_data[i].nelements    = end;
        }
                                            
        DEBUG_PRINT("Signaling work.\n");           // Then get to work!
        sync_res = pthread_barrier_wait( &work_start );
        if (!(sync_res == 0 || sync_res == PTHREAD_BARRIER_SERIAL_THREAD)) {
            DEBUG_PRINT("Error when opening work-start barrier [ERRNO: %d\n", sync_res);
        }
                                            
        DEBUG_PRINT("Waiting for workers.\n");      // Wait for them to finish
        sync_res = pthread_barrier_wait( &work_sync );
        if (!(sync_res == 0 || sync_res == PTHREAD_BARRIER_SERIAL_THREAD)) {
            DEBUG_PRINT("Error when opening the work-sync barrier [ERRNO: %d\n", sync_res);
        }

        for(i=0; i<worker_count-1;i++) {
            tstates[i] = tstates[worker_count-1];
        }
    }
    return BH_SUCCESS;
}

bh_error bh_ve_mcore_execute(bh_ir* bhir)
{
    //bh_intp count;
    bh_instruction* instr;
    bh_index  nelements;
    bh_error res = BH_SUCCESS;

    for(bh_intp i=0; i < bhir->ninstr; ++i) {
        instr = &bhir->instr_list[i];

        res = bh_vcache_malloc(instr);     // Allocate memory for operands
        if (res != BH_SUCCESS) {
            return res;
        }

        switch(instr->opcode) {              // Dispatch instruction
            case BH_NONE:                    // NOOP.
            case BH_DISCARD:
            case BH_SYNC:
                res = BH_SUCCESS;
                break;

            case BH_FREE:
                res = bh_vcache_free(instr);
                break;

            case BH_ADD_REDUCE:
            case BH_MULTIPLY_REDUCE:
            case BH_MINIMUM_REDUCE:
            case BH_MAXIMUM_REDUCE:
            case BH_LOGICAL_AND_REDUCE:
            case BH_BITWISE_AND_REDUCE:
            case BH_LOGICAL_OR_REDUCE:
            case BH_BITWISE_OR_REDUCE:
            case BH_LOGICAL_XOR_REDUCE:
            case BH_BITWISE_XOR_REDUCE:
				res = bh_compute_reduce_naive(instr);
            	break;

            case BH_USERFUNC:                // External libraries
                if(instr->userfunc->id == random_impl_id) {
                    res = random_impl(instr->userfunc, NULL);
                } else if(instr->userfunc->id == matmul_impl_id) {
                    res = matmul_impl(instr->userfunc, NULL);
                } else if(instr->userfunc->id == lu_impl_id) {
                    res = lu_impl(instr->userfunc, NULL);
                } else if(instr->userfunc->id == fft_impl_id) {
                    res = fft_impl(instr->userfunc, NULL);
                } else if(instr->userfunc->id == fft2_impl_id) {
                    res = fft2_impl(instr->userfunc, NULL);
                } else {
                    res = BH_USERFUNC_NOT_SUPPORTED;// Unsupported userfunc
                }
                break;

            default:                            // Built-in operations

                nelements = bh_nelements(instr->operand[0].ndim, instr->operand[0].shape);
                if (nelements < 1024*1024) {        // Do not bother threading...
                    res = bh_compute_apply_naive(instr);
                } else if (instr->operand[0].shape[instr->operand[0].ndim-1] < 100) {
                    res = bh_compute_apply_naive(instr);
                } else {                            // DO bother!
                    res = dispatch(instr, nelements);
                }
        }

        if (res != BH_SUCCESS) {    // Instruction failed
            break;
        }
    }

    return res;
}

bh_error bh_random( bh_userfunc *arg, void* ve_arg)
{
    return bh_compute_random( arg, ve_arg );
}

bh_error bh_matmul( bh_userfunc *arg, void* ve_arg)
{
    return bh_compute_matmul( arg, ve_arg );    
}

bh_error bh_ve_mcore_reg_func(char *fun, bh_intp *id) {

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
    else if(strcmp("bh_lu", fun) == 0)
    {
    	if (lu_impl == NULL)
    	{
			bh_component_get_func(myself, fun, &lu_impl);
			if (lu_impl == NULL)
				return BH_USERFUNC_NOT_SUPPORTED;

			lu_impl_id = *id;
			return BH_SUCCESS;			
        }
        else
        {
        	*id = lu_impl_id;
        	return BH_SUCCESS;
        }
    }
    else if(strcmp("bh_fft", fun) == 0)
    {
    	if (fft_impl == NULL)
    	{
			bh_component_get_func(myself, fun, &fft_impl);
			if (fft_impl == NULL)
				return BH_USERFUNC_NOT_SUPPORTED;

			fft_impl_id = *id;
			return BH_SUCCESS;			
        }
        else
        {
        	*id = fft_impl_id;
        	return BH_SUCCESS;
        }
    }
    else if(strcmp("bh_fft2", fun) == 0)
    {
    	if (fft2_impl == NULL)
    	{
			bh_component_get_func(myself, fun, &fft2_impl);
			if (fft2_impl == NULL)
				return BH_USERFUNC_NOT_SUPPORTED;

			fft2_impl_id = *id;
			return BH_SUCCESS;			
        }
        else
        {
        	*id = fft2_impl_id;
        	return BH_SUCCESS;
        }
    }
    
    return BH_USERFUNC_NOT_SUPPORTED;
}

#endif /*_POSIX_BARRIERS <= 0*/
