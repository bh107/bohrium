/*
 * Copyright 2011 Simon A. F. Lund <safl@safl.dk> 
 *
 * This file is part of cphVB.
 *
 * cphVB is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * cphVB is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with cphVB. If not, see <http://www.gnu.org/licenses/>.
 */
#include <cphvb.h>
#include <dispatch.h>
#include "cphvb_ve_mcore.h"

#include <queue>                                        //
#include <pthread.h>                                    // MULTICORE stuff
#include <semaphore.h>                                  //

#define MCORE_WORKERS 4                                 //

typedef struct worker_data {                            // Thread identity and control
    int     id;
    bool    keep_working;
} worker_data_t;

typedef struct job {                                    // Job specification
    cphvb_instruction *instr;
    int start;
    int end;
} job_t;

std::queue<job_t>   job_queue;                          // Job Queue, Lock, Barrier and Semaphore
pthread_mutex_t     job_queue_l;

sem_t               work;                               // Work syncronization
pthread_barrier_t   work_sync;

pthread_t           workers[MCORE_WORKERS];             // Worker-pool
worker_data_t       worker_data_array[MCORE_WORKERS];   // And their associated data

static void * worker(void *worker_arg) {
    
    worker_data_t   *my_data;
    job_t           my_job;
    int             sync_res;

    my_data = (worker_data_t *)worker_arg;              // Grab the thread argument
    
    printf("Worker %d - Started.\n", my_data->id);

    while( true ) {

        sem_wait( &work );

        pthread_mutex_lock( &job_queue_l );
        if (!job_queue.empty()) {

            my_job = job_queue.front();                 // Grab a job
            job_queue.pop();                            // Remove it from the queue
            pthread_mutex_unlock( &job_queue_l );

            if( my_job.instr != NULL ) {                // Computation job => Compute

                // Do the actual computation here...

                sync_res = pthread_barrier_wait( &work_sync );  // Wait for the others to finish
                if (!(sync_res == 0 || sync_res == PTHREAD_BARRIER_SERIAL_THREAD)) {
                    printf("Worker %d - Error synchronizing...\n", my_data->id);
                }

            } else {                                    // Pseudo-job => Exit
                printf("Worker %d - Exit job...\n", my_data->id);
                break;                                  // Exit...
            }
            
        } else {
            pthread_mutex_unlock( &job_queue_l );
        }
    
    }

    printf("Worker %d - Returning...\n", my_data->id);

    return NULL;

}

cphvb_error cphvb_ve_mcore_init(

    cphvb_intp      *opcode_count,
    cphvb_opcode    opcode_list[CPHVB_MAX_NO_OPERANDS],
    cphvb_intp      *datatype_count,
    cphvb_type      datatype_list[CPHVB_NO_TYPES]

) {
    cphvb_error res = CPHVB_SUCCESS;

    *opcode_count = 0;                                  // Announce supported opcodes
    for (cphvb_opcode oc = 0; oc <CPHVB_NO_OPCODES; ++oc) {
        opcode_list[*opcode_count] = oc;
        ++*opcode_count;
    }
    *datatype_count = 0;                                // Announce supported types
    for(cphvb_type ot = 0; ot <= CPHVB_FLOAT64; ++ot) {
        datatype_list[*datatype_count] = ot;
        ++*datatype_count;
    }
                                                        //
                                                        // Multicore initialization
                                                        //

                                                        // Mutex for job-queue protection
    if (pthread_mutex_init( &job_queue_l, NULL ) != 0) {
        return CPHVB_ERROR;
    }

    if (sem_init(&work, 0, 0) != 0) {                   // Semaphore for work signaling
        return CPHVB_ERROR;
    };                              
                                                        // Barrier for work syncronization
    if (pthread_barrier_init( &work_sync, NULL, MCORE_WORKERS) != 0) {
        return CPHVB_ERROR;
    }

    for(int i=0; i<MCORE_WORKERS; i++) {                // Worker-threads for job computation
        worker_data_array[i] = (worker_data_t){ i, true };
        if (pthread_create( &workers[i], NULL, worker, &worker_data_array[i] ) != 0) {
            res = CPHVB_ERROR;
            break;
        }
    }

    return res;
}

cphvb_error cphvb_ve_mcore_execute(

    cphvb_intp          instruction_count,
    cphvb_instruction   instruction_list[]

) {
   
    cphvb_error res = CPHVB_SUCCESS;

    for(cphvb_intp i=0; i<instruction_count; ++i) {

        cphvb_instruction   *instr  = &instruction_list[i];
        for(int w=0; w<MCORE_WORKERS; w++) {
            job_queue.push( (job_t){ instr, 0, 0 } );   // TODO: split the computation
        }

        for(int w=0; w<MCORE_WORKERS; w++) {
            sem_post( &work );
        }

    }

    // TODO: i guess i should wait here to see if workers succeeded...

    return res;

}

cphvb_error cphvb_ve_mcore_shutdown(void) {
   
    cphvb_error res;

    printf("Sending shutdown signals to workers...\n");
    pthread_mutex_lock( &job_queue_l );             // Send the shutdown job
    for(int i=0; i<MCORE_WORKERS; i++) {
        job_queue.push( (job_t){ NULL, 0, 0 } );
        sem_post( &work );
    }
    pthread_mutex_unlock( &job_queue_l );

    printf("Waiting for threads...\n");
    for(int i=0; i<MCORE_WORKERS; i++) {            // Wait for termination
        if (pthread_join( workers[i], NULL) != 0) {
            res = CPHVB_ERROR;
        }
    }
    printf("Threads joined!\n");
                                                    // Cleanup syncronization primitives
    sem_destroy( &work );
    pthread_barrier_destroy( &work_sync );
    pthread_mutex_destroy( &job_queue_l );

    return res;
}

