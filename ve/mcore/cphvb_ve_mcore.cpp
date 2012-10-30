/*
This file is part of cphVB and copyright (c) 2012 the cphVB team:
http://cphvb.bitbucket.org

cphVB is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as 
published by the Free Software Foundation, either version 3 
of the License, or (at your option) any later version.

cphVB is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the 
GNU Lesser General Public License along with cphVB. 

If not, see <http://www.gnu.org/licenses/>.
*/
#include <cphvb.h>
#include "cphvb_ve_mcore.h"
#include <cphvb_vcache.h>
#include <iostream>

#include <queue>                                        //
#include <pthread.h>                                    // MULTICORE libraries
#include <semaphore.h>                                  //

#if _POSIX_BARRIERS <= 0
#error This system does not support POSIX barriers
#else

static cphvb_component *myself = NULL;
static cphvb_userfunc_impl reduce_impl = NULL;
static cphvb_intp reduce_impl_id = 0;
static cphvb_userfunc_impl random_impl = NULL;
static cphvb_intp random_impl_id = 0;
static cphvb_userfunc_impl matmul_impl = NULL;
static cphvb_intp matmul_impl_id = 0;
static cphvb_userfunc_impl lu_impl = NULL;
static cphvb_intp lu_impl_id = 0;
static cphvb_userfunc_impl fft_impl = NULL;
static cphvb_intp fft_impl_id = 0;
static cphvb_userfunc_impl fft2_impl = NULL;
static cphvb_intp fft2_impl_id = 0;

//static cphvb_intp cphvb_ve_mcore_buffersizes = 0;
//static computeloop* cphvb_ve_mcore_compute_loops = NULL;
//static cphvb_tstate_naive* cphvb_ve_mcore_tstates = NULL;

static cphvb_intp block_size = 1000;
                                                        //
                                                        // MULTICORE datastructures and variables
                                                        //

typedef struct worker_data {                            // Thread identity and control
    int id;
    cphvb_computeloop_naive loop;
    cphvb_instruction *instr;
    cphvb_tstate_naive *state;
    cphvb_index nelements;
} worker_data_t;

static pthread_barrier_t   work_start;                  // Work synchronization using barrier
static pthread_barrier_t   work_sync;                   // Work synchronization using barrier

static pthread_t        worker[MCORE_MAX_WORKERS];          // Worker-pool
static worker_data_t    worker_data[MCORE_MAX_WORKERS];     // And their associated data

static int worker_count = MCORE_WORKERS;
static cphvb_tstate_naive tstates[MCORE_MAX_WORKERS];

static void* job(void *worker_arg)
{
    int sync_res;
    worker_data_t *my_job = (worker_data_t*)worker_arg;     // Grab the thread argument

    DEBUG_PRINT("Worker %d - Started.\n", my_job->id);

    while( true ) {

        sync_res = pthread_barrier_wait( &work_start );     // Wait for work
        if (!(sync_res == 0 || sync_res == PTHREAD_BARRIER_SERIAL_THREAD)) {
            DEBUG_PRINT("Worker %d - Error waiting for work! [ERRNO: %d]\n", my_job->id, sync_res);
        }

    
        if ( my_job->instr != NULL ) {                      // Got a job

            if ( my_job->instr->opcode == CPHVB_USERFUNC ) {      // userfunc

                cphvb_compute_apply_naive( my_job->instr );

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

cphvb_error cphvb_ve_mcore_init(cphvb_component *self)
{
    char *env;
    int i;
    cphvb_error res = CPHVB_SUCCESS;
    myself = self;                              // Assign config container.

    env = getenv("CPHVB_VE_MCORE_BLOCKSIZE");         // Override block_size from ENVVAR
    if (env != NULL) {
        block_size = atoi(env);
    }
    if (block_size <= 0) {                      // Verify it
        fprintf(stderr, "CPHVB_VE_MCORE_BLOCKSIZE (%ld) should be greater than zero!\n", (long)block_size);
        return CPHVB_ERROR;
    }

    env = getenv("CPHVB_VE_MCORE_NTHREADS");          // Override worker_count with ENVVAR
    if (env != NULL) {
        worker_count = atoi(env);
    }

    if (worker_count > MCORE_MAX_WORKERS) {     // Verify worker count
        fprintf(stderr,"CPHVB_VE_MCORE_NTHREADS capped to %i.\n", MCORE_MAX_WORKERS);
        worker_count = MCORE_MAX_WORKERS;
    } else if (worker_count < 1) {
        fprintf(stderr,"CPHVB_VE_MCORE_NTHREADS capped to default %i.\n", MCORE_WORKERS);
        worker_count = MCORE_WORKERS;
    }

    cphvb_vcache_init( 10 );                            // Malloc-cache initialization

                                                        //
                                                        // Multicore initialization
                                                        //

                                                        // Barriers for work syncronization
    if (pthread_barrier_init( &work_start, NULL, worker_count+1) != 0) {
        return CPHVB_ERROR;
    }
    if (pthread_barrier_init( &work_sync, NULL, worker_count+1) != 0) {
        return CPHVB_ERROR;
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
        worker_data[i] = (worker_data_t){ i, NULL, NULL, NULL, 0 };
#endif
        if (pthread_create( &worker[i], NULL, job, &worker_data[i] ) != 0) {
            res = CPHVB_ERROR;
            break;
        }
    }

    return res;

}

cphvb_error cphvb_ve_mcore_shutdown( void )
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
    pthread_barrier_destroy( &work_start );
    pthread_barrier_destroy( &work_sync );

    // De-allocate the malloc-cache
    cphvb_vcache_clear();
    cphvb_vcache_delete();


    return CPHVB_SUCCESS;
}

inline cphvb_error dispatch( cphvb_instruction* instr, cphvb_index nelements) {

    int sync_res;
    cphvb_computeloop_naive loop;
    cphvb_intp i;
    cphvb_index  last_dim, start, end, size;

    loop     = cphvb_compute_get_naive( instr );
    last_dim = instr->operand[0]->ndim-1;
    size     = nelements / worker_count;

    for (i=0; i<worker_count;i++)       // tstate = (0, 0, 0, ..., 0)
        cphvb_tstate_reset_naive( &tstates[i] );  
    while(tstates[worker_count-1].cur_e < nelements) {

        for(i=0;i<worker_count;i++) {   // Setup workers

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

    return CPHVB_SUCCESS;

}

cphvb_error cphvb_ve_mcore_execute( cphvb_intp instruction_count, cphvb_instruction* instruction_list )
{
    cphvb_intp count;
    cphvb_instruction* inst;
    cphvb_index  nelements;
    cphvb_error res;

    for(count=0; count < instruction_count; count++)
    {
        inst = &instruction_list[count];

        if(inst->status == CPHVB_SUCCESS)     // SKIP instruction
        {
            continue;
        }

        res = cphvb_vcache_malloc( inst );      // Allocate memory for operands
        if ( res != CPHVB_SUCCESS ) {
            return res;
        }

        switch(inst->opcode)                    // Dispatch instruction
        {
            case CPHVB_NONE:                    // NOOP.
            case CPHVB_DISCARD:
            case CPHVB_SYNC:
                inst->status = CPHVB_SUCCESS;
                break;

            case CPHVB_FREE:

                inst->status = cphvb_vcache_free( inst );
                break;

            case CPHVB_USERFUNC:                // External libraries

                if(inst->userfunc->id == reduce_impl_id)
                {
                    inst->status = reduce_impl(inst->userfunc, NULL);
                }
                else if(inst->userfunc->id == random_impl_id)
                {
                    inst->status = random_impl(inst->userfunc, NULL);
                }
                else if(inst->userfunc->id == matmul_impl_id)
                {
                    inst->status = matmul_impl(inst->userfunc, NULL);
                }
                else if(inst->userfunc->id == lu_impl_id)
                {
                    inst->status = lu_impl(inst->userfunc, NULL);
                }
                else if(inst->userfunc->id == fft_impl_id)
                {
                    inst->status = fft_impl(inst->userfunc, NULL);
                }
                else if(inst->userfunc->id == fft2_impl_id)
                {
                    inst->status = fft2_impl(inst->userfunc, NULL);
                }
                else                            // Unsupported userfunc
                {
                    inst->status = CPHVB_USERFUNC_NOT_SUPPORTED;
                }

                break;

            default:                            // Built-in operations

                nelements   = cphvb_nelements( inst->operand[0]->ndim, inst->operand[0]->shape );

                if (nelements < 1024*1024) {        // Do not bother threading...
                    inst->status = cphvb_compute_apply_naive( inst );
                } else {                            // DO bother!
                    inst->status = dispatch( inst, nelements );
                }
                
        }

        if (inst->status != CPHVB_SUCCESS)    // Instruction failed
        {
            break;
        }

    }

    if (count == instruction_count) {
        return CPHVB_SUCCESS;
    } else {
        return CPHVB_PARTIAL_SUCCESS;
    }

}



cphvb_error cphvb_random( cphvb_userfunc *arg, void* ve_arg)
{
    return cphvb_compute_random( arg, ve_arg );
}

cphvb_error cphvb_matmul( cphvb_userfunc *arg, void* ve_arg)
{
    return cphvb_compute_matmul( arg, ve_arg );
    
}

/**
 * cphvb_compute_reduce
 *
 * Implementation of the user-defined funtion "reduce".
 * Note that we follow the function signature defined by cphvb_userfunc_impl.
 *
 * This function is functionaly equivalent to cphvb_compute_reduce,
 * the difference is in the implementation... it should be able to run
 * faster when utilizing multiple threads of execution.
 *
 */
cphvb_error cphvb_reduce( cphvb_userfunc *arg, void* ve_arg )
{
    cphvb_reduce_type *a = (cphvb_reduce_type *) arg;
    cphvb_instruction inst;
    cphvb_error err;
    cphvb_intp i, j, step, axis_size;
    cphvb_array *out, *in, tmp;
    //cphvb_index nelements;

    if (cphvb_operands(a->opcode) != 3) {
        fprintf(stderr, "ERR: Reduce only support binary operations.\n");
        return CPHVB_ERROR;
    }

	if (cphvb_base_array(a->operand[1])->data == NULL) {
        fprintf(stderr, "ERR: Reduce called with input set to null.\n");
        return CPHVB_ERROR;
	}
                                                // Make sure that the array memory is allocated.
    if (cphvb_data_malloc(a->operand[0]) != CPHVB_SUCCESS) {
        return CPHVB_OUT_OF_MEMORY;
    }
    
    out = a->operand[0];
    in  = a->operand[1];
    
    tmp         = *in;                          // Copy the input-array meta-data
    tmp.base    = cphvb_base_array(in);
    tmp.start   = in->start;

    step = in->stride[a->axis];
    j=0;
    for(i=0; i<in->ndim; ++i) {                 // Remove the 'axis' dimension from in
        if(i != a->axis) {
            tmp.shape[j]    = in->shape[i];
            tmp.stride[j]   = in->stride[i];
            ++j;
        }
    }
    if (tmp.ndim > 1) {                         // NOTE:    It just seems strange that it should
        tmp.ndim--;                             //          be able to have 0 dimensions...
    }
    
    inst.status = CPHVB_INST_PENDING;           // We copy the first element to the output.
    inst.opcode = CPHVB_IDENTITY;
    inst.operand[0] = out;
    inst.operand[1] = &tmp;
    inst.operand[2] = NULL;

    //nelements   = cphvb_nelements( inst.operand[0]->ndim, inst.operand[0]->shape );
    //err         = dispatch( &inst, nelements );
    err = cphvb_compute_apply_naive( &inst );
    if (err != CPHVB_SUCCESS) {
        return err;
    }
    tmp.start += step;

    inst.status = CPHVB_INST_PENDING;           // Reduce over the 'axis' dimension.
    inst.opcode = a->opcode;                    // NB: the first element is already handled.
    inst.operand[0] = out;
    inst.operand[1] = out;
    inst.operand[2] = &tmp;
    
    axis_size = in->shape[a->axis];

    for(i=1; i<axis_size; ++i) {                // Execute!
        err = cphvb_compute_apply_naive( &inst );
        //nelements   = cphvb_nelements( inst.operand[0]->ndim, inst.operand[0]->shape );
        //err         = dispatch( &inst, nelements );
        //err         = dispatch( &inst, nelements );
        if (err != CPHVB_SUCCESS) {
            return err;
        }
        tmp.start += step;
    }

    return CPHVB_SUCCESS;
}

cphvb_error cphvb_ve_mcore_reg_func(char *fun, cphvb_intp *id) {

    if(strcmp("cphvb_reduce", fun) == 0)
    {
    	if (reduce_impl == NULL)
    	{
			cphvb_component_get_func(myself, fun, &reduce_impl);
			if (reduce_impl == NULL)
				return CPHVB_USERFUNC_NOT_SUPPORTED;

			reduce_impl_id = *id;
			return CPHVB_SUCCESS;			
        }
        else
        {
        	*id = reduce_impl_id;
        	return CPHVB_SUCCESS;
        }
    }
    else if(strcmp("cphvb_random", fun) == 0)
    {
    	if (random_impl == NULL)
    	{
			cphvb_component_get_func(myself, fun, &random_impl);
			if (random_impl == NULL)
				return CPHVB_USERFUNC_NOT_SUPPORTED;

			random_impl_id = *id;
			return CPHVB_SUCCESS;			
        }
        else
        {
        	*id = random_impl_id;
        	return CPHVB_SUCCESS;
        }
    }
    else if(strcmp("cphvb_matmul", fun) == 0)
    {
    	if (matmul_impl == NULL)
    	{
			cphvb_component_get_func(myself, fun, &matmul_impl);
			if (matmul_impl == NULL)
				return CPHVB_USERFUNC_NOT_SUPPORTED;

			matmul_impl_id = *id;
			return CPHVB_SUCCESS;			
        }
        else
        {
        	*id = matmul_impl_id;
        	return CPHVB_SUCCESS;
        }
    }
    else if(strcmp("cphvb_lu", fun) == 0)
    {
    	if (lu_impl == NULL)
    	{
			cphvb_component_get_func(myself, fun, &lu_impl);
			if (lu_impl == NULL)
				return CPHVB_USERFUNC_NOT_SUPPORTED;

			lu_impl_id = *id;
			return CPHVB_SUCCESS;			
        }
        else
        {
        	*id = lu_impl_id;
        	return CPHVB_SUCCESS;
        }
    }
    else if(strcmp("cphvb_fft", fun) == 0)
    {
    	if (fft_impl == NULL)
    	{
			cphvb_component_get_func(myself, fun, &fft_impl);
			if (fft_impl == NULL)
				return CPHVB_USERFUNC_NOT_SUPPORTED;

			fft_impl_id = *id;
			return CPHVB_SUCCESS;			
        }
        else
        {
        	*id = fft_impl_id;
        	return CPHVB_SUCCESS;
        }
    }
    else if(strcmp("cphvb_fft2", fun) == 0)
    {
    	if (fft2_impl == NULL)
    	{
			cphvb_component_get_func(myself, fun, &fft2_impl);
			if (fft2_impl == NULL)
				return CPHVB_USERFUNC_NOT_SUPPORTED;

			fft2_impl_id = *id;
			return CPHVB_SUCCESS;			
        }
        else
        {
        	*id = fft2_impl_id;
        	return CPHVB_SUCCESS;
        }
    }
    
    return CPHVB_USERFUNC_NOT_SUPPORTED;
}


/*
inline cphvb_error block_execute( cphvb_instruction* instr, cphvb_intp start, cphvb_intp end) {

    cphvb_intp i, k, w;
    int             sync_res;

    cphvb_index last_dim = instr[start].operand[0]->ndim-1;

    //Make sure we have enough space
    if ((end - start + 1) > cphvb_ve_mcore_buffersizes) 
    {
    	cphvb_intp mcount = (end - start + 1);

    	//We only work in multiples of 1000
    	if (mcount % 1000 != 0)
    		mcount = (mcount + 1000) - (mcount % 1000);

    	//Make sure we re-allocate on error
    	cphvb_ve_mcore_buffersizes = 0;
    	
    	if (cphvb_ve_mcore_compute_loops != NULL) {
    		free(cphvb_ve_mcore_compute_loops);
    		free(cphvb_ve_mcore_tstates);
    		cphvb_ve_mcore_compute_loops = NULL;
    		cphvb_ve_mcore_tstates = NULL;
    	}
    	
    	cphvb_ve_mcore_compute_loops = (computeloop*)malloc(sizeof(computeloop) * mcount);
        cphvb_ve_mcore_tstates = (cphvb_tstate*)malloc(sizeof(cphvb_tstate)*mcount);
    	
    	if (cphvb_ve_mcore_compute_loops == NULL)
    		return CPHVB_OUT_OF_MEMORY;
    	if (cphvb_ve_mcore_tstates == NULL)
    		return CPHVB_OUT_OF_MEMORY;
    	
    	cphvb_ve_mcore_buffersizes = mcount;
    }
    
    computeloop* compute_loops = cphvb_ve_mcore_compute_loops;
    cphvb_tstate* states = cphvb_ve_mcore_tstates;
    cphvb_index  nelements, trav_start=0, trav_end=0;
    
    for(i=0; i<=end-start;i++)                      // Reset traversal coordinates
        cphvb_tstate_reset( &states[i] );

    for(i=start, k=0; i <= end; i++,k++)            // Get the compute-loops
    {
        compute_loops[k] = cphvb_compute_get( &instr[i] );
    }
                                                    // Block-size split
    nelements = cphvb_nelements( instr[start].operand[0]->ndim, instr[start].operand[0]->shape );

    if (nelements < (1024)) {
        for(i=start, k=0; i <= end; i++, k++) {
            compute_loops[k]( &instr[i], &states[k], 0 );
        }
    } else {

        for(i=start, k=0; i <= end; i++, k++)
        {
            while(nelements>trav_end)
            {
                trav_start  = trav_end;
                trav_end    += block_size;
                if (trav_end > nelements)
                    trav_end = nelements;

                for(w=0; w<worker_count; w++) {     // Setup data structures for workers
                    states[k].coord[last_dim] = trav_start;

                    worker_data[w].loop         = compute_loops[k];
                    worker_data[w].instr        = &instr[i];
                    worker_data[w].state        = &states[k];
                    worker_data[w].nelements    = trav_end;
                }
                                                    // Signal work
                DEBUG_PRINT("Signaling work.\n");
                sync_res = pthread_barrier_wait( &work_start );
                if (!(sync_res == 0 || sync_res == PTHREAD_BARRIER_SERIAL_THREAD)) {
                    DEBUG_PRINT("Error when opening work-start barrier [ERRNO: %d\n", sync_res);
                }
                                                    // Wait for them to finish
                DEBUG_PRINT("Waiting for workers.\n");
                sync_res = pthread_barrier_wait( &work_sync );
                if (!(sync_res == 0 || sync_res == PTHREAD_BARRIER_SERIAL_THREAD)) {
                    DEBUG_PRINT("Error when opening the work-sync barrier [ERRNO: %d\n", sync_res);
                }

            }
        }
    }

    for(i=start; i <= end; i++)                     // Set instruction status
    {
        instr[i].status = CPHVB_SUCCESS;
    }
    
    return CPHVB_SUCCESS;

}

cphvb_error cphvb_ve_mcore_execute( cphvb_intp instruction_count, cphvb_instruction* instruction_list )
{
    cphvb_intp cur_index, nops, i, j;
    cphvb_instruction *inst, *binst;

    cphvb_intp bin_start, bin_end, bin_size;
    cphvb_intp bundle_start, bundle_end, bundle_size;

    for(cur_index=0; cur_index < instruction_count; cur_index++)
    {
        inst = &instruction_list[cur_index];

        if(inst->status == CPHVB_SUCCESS)       // SKIP instruction
        {
            continue;
        }

        nops = cphvb_operands(inst->opcode);    // Allocate memory for operands
        for(i=0; i<nops; i++)
        {
            if (!cphvb_is_constant(inst->operand[i]))
            {
                if (cphvb_data_malloc(inst->operand[i]) != CPHVB_SUCCESS)
                {
                    return CPHVB_OUT_OF_MEMORY; // EXIT
                }
            }

        }

        switch(inst->opcode)                    // Dispatch instruction
        {
            case CPHVB_NONE:                    // NOOP.
            case CPHVB_DISCARD:
            case CPHVB_SYNC:
                inst->status = CPHVB_SUCCESS;
                break;

            case CPHVB_USERFUNC:                // External libraries

                if(inst->userfunc->id == reduce_impl_id)
                {
                    inst->status = reduce_impl(inst->userfunc, NULL);
                }
                else if(inst->userfunc->id == random_impl_id)
                {
                    inst->status = random_impl(inst->userfunc, NULL);
                }
                else if(inst->userfunc->id == matmul_impl_id)
                {
                    inst->status = matmul_impl(inst->userfunc, NULL);
                }
                else                            // Unsupported userfunc
                {
                    inst->status = CPHVB_USERFUNC_NOT_SUPPORTED;
                }

                break;

            default:                                            // Built-in operations
                                                                // -={[ BINNING ]}=-
                bin_start   = cur_index;                        // Count built-ins and their start/end indexes and allocate memory for them.
                bin_end     = cur_index;
                for(j=bin_start+1; j<instruction_count; j++)    
                {
                    binst = &instruction_list[j];               // EXIT: Stop if instruction is NOT built-in
                    if ((binst->opcode == CPHVB_NONE) || (binst->opcode == CPHVB_DISCARD) || (binst->opcode == CPHVB_SYNC) ||(binst->opcode == CPHVB_USERFUNC) ) {
                        break;
                    }

                    bin_end++;                                  // The "end" index

                    nops = cphvb_operands(binst->opcode);       // The memory part...
                    for(i=0; i<nops; i++)
                    {
                        if (!cphvb_is_constant(binst->operand[i]))
                        {
                            if (cphvb_data_malloc(binst->operand[i]) != CPHVB_SUCCESS)
                            {
                                return CPHVB_OUT_OF_MEMORY;     // EXIT
                            }
                        }

                    }

                }
                bin_size = bin_end - bin_start +1;              // The counting part

                                                                // -={[ BUNDLING ]}=-
                bundle_size     = (bin_size > 1) ? cphvb_inst_bundle( instruction_list, bin_start, bin_end ) : 1;
                bundle_start    = bin_start;
                bundle_end      = bundle_start + bundle_size-1;

                block_execute( instruction_list, bundle_start, bundle_end );
                cur_index += bundle_size-1;
        }

        if (inst->status != CPHVB_SUCCESS)    // Instruction failed
        {
            break;
        }

    }

    if (cur_index == instruction_count) {
        return CPHVB_SUCCESS;
    } else {
        return CPHVB_PARTIAL_SUCCESS;
    }

}


*/

#endif /*_POSIX_BARRIERS <= 0*/
