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

//Thread function and it's ID.
typedef struct
{
    int myid;
    cphvb_intp nblocks;
    cphvb_intp nthds;
    cphvb_intp size;
    cphvb_instruction** inst_bundle;
    traverse_ptr *traverses;
}thd_id;

void *thd_do(void *msg)
{
    thd_id *id = (thd_id *) msg;
    int myid = id->myid;
    cphvb_intp size = id->size;
    cphvb_intp nblocks = id->nblocks;
    cphvb_intp nthds = id->nthds;
    cphvb_instruction** inst_bundle = id->inst_bundle;
    traverse_ptr *traverses = id->traverses;

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

        if(thd_inst[0].operand[0]->shape[0] <= 0)
            break;//We a finished.

        //Dispatch a block.
        for(cphvb_intp j=0; j<size; ++j)
        {
            cphvb_instruction *inst = &thd_inst[j];
            inst->status = traverses[j](inst);
            if(inst->status != CPHVB_SUCCESS)
            {
                //ret = CPHVB_PARTIAL_SUCCESS;
                start = length;//Force a complete exit.
                break;
            }
        }
    }

    return NULL;
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

    //Handle the blocks.
    //We will use OpenMP to parallelize the computation.
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
    thd_id ids[32];
    pthread_t tid[32];
    for(cphvb_intp i=0; i<nthds; ++i)
    {
        ids[i].myid        = i;
        ids[i].nblocks     = nblocks;
        ids[i].nthds       = nthds;
        ids[i].size        = size;
        ids[i].inst_bundle = inst_bundle;
        ids[i].traverses   = traverses;
        pthread_create(&tid[i], NULL, thd_do, (void *) (&ids[i]));
    }
    //Stop threads.
    for(cphvb_intp i=0; i<nthds; ++i)
        pthread_join(tid[i], NULL);


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
