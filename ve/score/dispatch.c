#include <cphvb.h>
#include "get_traverse.hpp"
#include "dispatch.h"
#include <assert.h>
#include <pthread.h>

#ifdef _OPENMP
    #include <omp.h>
#endif

//The data type of cphvb_array extended with temporary info.
typedef struct
{
    CPHVB_ARRAY_HEAD
    //Saved original values
    cphvb_index      org_start;
    cphvb_index      org_shape[CPHVB_MAXDIM];
}dispatch_ary;

//Dispatch the instruction.
cphvb_error dispatch( cphvb_instruction *instr ) {

    traverse_ptr traverser = get_traverse( instr );

    if (traverser == NULL) {
        return CPHVB_ERROR;
    } else {
        return traverser( instr );
    }

}

//Returns the offset based on the current block.
inline cphvb_intp get_offset(cphvb_intp block, cphvb_instruction *inst,
                             cphvb_intp nblocks)
{
    if(block == 0)
        return 0;

    //We compute the offset based on the output operand.
    dispatch_ary *ary = (dispatch_ary*) inst->operand[0];

    return ary->org_shape[0] / nblocks * block + //Whole blocks
           ary->org_shape[0] % nblocks;//The reminder.
}

//Returns the shape based on the current block.
inline cphvb_intp get_shape(cphvb_intp block, cphvb_instruction *inst,
                            cphvb_intp nblocks)
{
    dispatch_ary *ary = (dispatch_ary*) inst->operand[0];

    //We block over the most significant dimension
    //and the first block gets the reminder.
    if(block == 0)
    {
        return ary->org_shape[0] / nblocks +
               ary->org_shape[0] % nblocks;
    }
    else
    {
        return ary->org_shape[0] / nblocks;
    }
}

//Shared thread variables.
pthread_cond_t  thd_cond_wait     = PTHREAD_COND_INITIALIZER;
pthread_cond_t  thd_cond_finished = PTHREAD_COND_INITIALIZER;
pthread_cond_t  thd_cond_restart  = PTHREAD_COND_INITIALIZER;
pthread_mutex_t thd_mutex         = PTHREAD_MUTEX_INITIALIZER;
int thd_wait         = 1;//0==start,1==wait
int thd_restart      = 1;//0==restart,1==wait
int thd_finish_count = 0;//Whan it reaches nthd we are finished.
int thd_exit         = 0;//1==exit,0=running
cphvb_intp thd_nblocks;
cphvb_intp thd_nthds;
cphvb_intp thd_size;
cphvb_instruction** thd_inst_bundle;
traverse_ptr *thd_traverses;
//The thread function.
void *thd_do(void *msg)
{
    while(1)
    {
        //Wait for start signal.
        pthread_mutex_lock(&thd_mutex);
        while(thd_wait)
            pthread_cond_wait(&thd_cond_wait,&thd_mutex);
        pthread_mutex_unlock(&thd_mutex);

        if(thd_exit)
            break;//We should exit.

        cphvb_intp myid = ((cphvb_intp)msg);
        cphvb_intp size = thd_size;
        cphvb_intp nblocks = thd_nblocks;
        cphvb_intp nthds = thd_nthds;
        cphvb_instruction** inst_bundle = thd_inst_bundle;
        traverse_ptr *traverses = thd_traverses;

        //We will not block a single instruction into more blocks
        //than threads.
        if(thd_size == 1)
            if(nblocks > nthds)
                nblocks = nthds;

        cphvb_intp length = nblocks / nthds; // Find this thread's length of work.
        cphvb_intp start = myid * length;    // Find this thread's start block.
        if(myid == nthds-1)
            length += nblocks % nthds;       // The last thread gets the reminder.

        //Clone the instruction and make new views of all operands.
        cphvb_instruction thd_inst[CPHVB_MAX_NO_INST];
        cphvb_array ary_stack[CPHVB_MAX_NO_INST*CPHVB_MAX_NO_OPERANDS];
        cphvb_intp ary_stack_count=0;
        for(cphvb_intp j=0; j<size; ++j)
        {
            thd_inst[j] = *inst_bundle[j];
            for(cphvb_intp i=0; i<cphvb_operands(inst_bundle[j]->opcode); ++i)
            {
                cphvb_array *ary_org = inst_bundle[j]->operand[i];
                cphvb_array *ary = &ary_stack[ary_stack_count++];
                *ary = *ary_org;

                if(ary_org->base == NULL)//base array
                {
                    ary->base = ary_org;
                }

                thd_inst[j].operand[i] = ary;//Save the operand.
             }
        }

        //Handle one block at a time.
        for(cphvb_intp b=start; b<start+length; ++b)
        {
            //Update the operands to match the current block.
            for(cphvb_intp j=0; j<size; ++j)
            {
                cphvb_instruction *inst = &thd_inst[j];
                for(cphvb_intp i=0; i<cphvb_operands(inst->opcode); ++i)
                {
                    dispatch_ary *ary = (dispatch_ary*) inst->operand[i];
                    ary->start = ary->org_start + ary->stride[0] *
                                 get_offset(b, inst, nblocks);
                    ary->shape[0] = get_shape(b, inst, nblocks);
                }
            }

            //Dispatch a block.
            for(cphvb_intp j=0; j<size; ++j)
            {
                cphvb_instruction *inst = &thd_inst[j];
                if(inst->operand[0]->shape[0] <= 0)
                    break;//Nothing to do.

                inst->status = traverses[j](inst);
                if(inst->status != CPHVB_SUCCESS)
                {
                    //ret = CPHVB_PARTIAL_SUCCESS;
                    start = length;//Force a complete exit.
                    break;
                }
            }
        }

        //Signal that we are finished.
        pthread_mutex_lock(&thd_mutex);
        if(++thd_finish_count == thd_nthds)//We are the last to finish.
            pthread_cond_signal(&thd_cond_finished);
        pthread_mutex_unlock(&thd_mutex);

        //Wait for restart signal.
        pthread_mutex_lock(&thd_mutex);
        while(thd_restart)
            pthread_cond_wait(&thd_cond_restart,&thd_mutex);
        pthread_mutex_unlock(&thd_mutex);
    }
    pthread_exit(NULL);
}


//Dispatch the bundle of instructions.
cphvb_error dispatch_bundle(cphvb_instruction** inst_bundle,
                            cphvb_intp size,
                            cphvb_intp nblocks)
{
    cphvb_error ret = CPHVB_SUCCESS;
    cphvb_intp nthds = 1;

    //Get all traverse function -- one per instruction.
    traverse_ptr traverses[CPHVB_MAX_NO_INST];
    for(cphvb_intp j=0; j<size; ++j)
    {
        traverses[j] = get_traverse( inst_bundle[j] );
        if(traverses[j] == NULL)
        {
            inst_bundle[j]->status = CPHVB_OUT_OF_MEMORY;
            ret = CPHVB_INST_NOT_SUPPORTED;
            goto finish;
        }
    }
    //Save the original array information.
    for(cphvb_intp j=0; j<size; ++j)
    {
        cphvb_instruction *inst = inst_bundle[j];
        for(cphvb_intp i=0; i<cphvb_operands(inst->opcode); ++i)
        {
            dispatch_ary *ary = (dispatch_ary*) inst->operand[i];
            ary->org_start = ary->start;
            memcpy(ary->org_shape, ary->shape, ary->ndim * sizeof(cphvb_index));
        }
    }
    //Make sure that all array-data is allocated.
    for(cphvb_intp j=0; j<size; ++j)
    {
        cphvb_instruction *inst = inst_bundle[j];
        for(cphvb_intp i=0; i<cphvb_operands(inst->opcode); ++i)
        {
            if(cphvb_data_malloc(inst->operand[i]) != CPHVB_SUCCESS)
            {
                inst->status = CPHVB_OUT_OF_MEMORY;
                ret = CPHVB_PARTIAL_SUCCESS;
                goto finish;
            }
        }
    }

    //We divide the blocks between the threads.
    if(nblocks > 1) //Find number of threads to use.
    {
        char *env = getenv("CPHVB_NUM_THREADS");
        if(env != NULL)
            nthds = atoi(env);

        if(nthds > nblocks)
            nthds = nblocks;//Minimum one block per thread.
        if(nthds > 32)
        {
            printf("CPHVB_NUM_THREADS greater than 32!\n");
            nthds = 32;//MAX 32 thds.
        }
    }

    //Start threads.
    pthread_t tid[32];
    thd_nblocks      = nblocks;
    thd_nthds        = nthds;
    thd_size         = size;
    thd_inst_bundle  = inst_bundle;
    thd_traverses    = traverses;

    //Create all threads.
    for(cphvb_intp i=0; i<nthds; ++i)
        pthread_create(&tid[i], NULL, thd_do, (void *) (i));

    //Start all threads.
    pthread_mutex_lock(&thd_mutex);
    thd_wait = 0;//False is the start signal.
    thd_finish_count = 0;//Reset the finish count.
    thd_restart = 1;//Reset the restart signal.
    pthread_cond_broadcast(&thd_cond_wait);
    pthread_mutex_unlock(&thd_mutex);

    //Wait for them to finish.
    pthread_mutex_lock(&thd_mutex);
    while(thd_finish_count < thd_nthds)
        pthread_cond_wait(&thd_cond_finished,&thd_mutex);
    pthread_mutex_unlock(&thd_mutex);

    //Restart all threads (making sure they are ready for next time).
    pthread_mutex_lock(&thd_mutex);
    thd_wait = 1;        //Reset the wait signal.
    thd_restart = 0;     //Set the restart signal.
    pthread_cond_broadcast(&thd_cond_restart);
    pthread_mutex_unlock(&thd_mutex);

    //Kill all threads.
    pthread_mutex_lock(&thd_mutex);
    thd_wait = 0;//Set the start signal.
    thd_exit = 1;//Set the exit signal.
    pthread_cond_broadcast(&thd_cond_wait);
    pthread_mutex_unlock(&thd_mutex);

    //Stop threads.
    for(cphvb_intp i=0; i<nthds; ++i)
        pthread_join(tid[i], NULL);

    //Reset signals.
    thd_wait         = 1;//0==start,1==wait
    thd_restart      = 1;//0==restart,1==wait
    thd_finish_count = 0;//Whan it reaches nthd we are finished.
    thd_exit         = 0;//1==exit,0=running

finish:
/*
    This is not needed anymore.
    //Restore the original arrays.
    for(cphvb_intp j=0; j<size; ++j)
    {
        cphvb_instruction *inst = inst_bundle[j];
        for(cphvb_intp i=0; i<cphvb_operands(inst->opcode); ++i)
        {
            dispatch_ary *ary = (dispatch_ary*) inst->operand[i];
            ary->start = ary->org_start;
            memcpy(ary->shape, ary->org_shape, ary->ndim * sizeof(cphvb_index));
        }
    }
*/
    return ret;
}
