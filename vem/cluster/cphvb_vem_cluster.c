/*
 * Copyright 2011 Mads R. B. Kristensen <madsbk@gmail.com>
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
#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include "private.h"
#include "message.h"
#include "handle.h"
#include <cphvb_vem.h>
#include <cphvb_vem_cluster.h>

//Temporary memory used for the communication message.
char message_tmp_mem[CLUSTER_MSG_SIZE];
void *msg_mem = message_tmp_mem;

//The VE info.
cphvb_support ve_support;

/* Initialize the VEM.
 * This is a collective operation.
 *
 * @return Error codes (CPHVB_SUCCESS, CPHVB_OUT_OF_MEMORY)
 */
cphvb_error cphvb_vem_cluster_init(void)
{
    int provided;

    //We make use of MPI_Init_thread even though we only ask for
    //a MPI_THREAD_SINGLE level thread-safety because MPICH2 only
    //supports MPICH_ASYNC_PROGRESS when MPI_Init_thread is used.
    //Note that when MPICH_ASYNC_PROGRESS is defined the thread-safety
    //level will automatically be set to MPI_THREAD_MULTIPLE.
    MPI_Init_thread(NULL, NULL, MPI_THREAD_SINGLE, &provided);

    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldsize);

    printf("cphvb_vem_cluster_init -- rank %d of %d\n", myrank, worldsize);
/*
    //Allocate buffers.
    workbuf = malloc(DNPY_WORK_BUFFER_MAXSIZE);
    workbuf_nextfree = workbuf;
    assert(workbuf != NULL);
*/
    return CPHVB_SUCCESS;
}

/* From this point on the master will continue with the pyton code
 * and the slaves will stay in C.
 * This only makes sense when combined with VEM_CLUSTER.
 * And this is a collective operation.
 * @return Zero when this is a master.
 */
cphvb_intp cphvb_vem_cluster_master_slave_split(void)
{
    if(myrank == 0)
    {
/*
        int tmpsizes[CPHVB_MAXDIM*CPHVB_MAXDIM];
        cluster_msg msg;
        //Check for user-defined block size.
        char *env = getenv("CLUSTER_BLOCKSIZE");
        if(env == NULL)
            blocksize = CLUSTER_BLOCKSIZE;
        else
            blocksize = atoi(env);

        if(blocksize <= 0)
        {
            fprintf(stderr, "User-defined blocksize must be greater "
                            "than zero\n");
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
        //Send block size to clients.

        msg->type = CLUSTER_INIT_BLOCKSIZE;
        msg[1] = blocksize;
        msg[2] = CLUSTER_MSG_END;
//        msg2slaves(msg, 3 * sizeof(cphvb_intp));
        #ifdef DEBUG
            printf("Rank 0 received msg: ");
        #endif
//        do_INIT_BLOCKSIZE(msg[1]);

        //Check for user-defined process grid.
        //The syntax used is: ndims:dim:size;
        //E.g. DNPY_PROC_SIZE="2:2:4;3:3:2" which means that array
        //with two dimensions should, at its second dimension, have
        //a size of four etc.
        memset(tmpsizes, 0, CPHVB_MAXDIM*CPHVB_MAXDIM*sizeof(int));
        env = getenv("CLUSTER_PROC_GRID");
        if(env != NULL)
        {
            char *res = strtok(env, ";");
            while(res != NULL)
            {
                char *size_ptr;
                int dsize = 0;
                int dim = 0;
                int ndims = strtol(res, &size_ptr, 10);
                if(size_ptr != '\0')
                    dim = strtol(size_ptr+1, &size_ptr, 10);
                if(size_ptr != '\0')
                    dsize = strtol(size_ptr+1, NULL, 10);
                //Make sure the input is valid.
                if(dsize <= 0 || dim <= 0 || dim > ndims ||
                   ndims <= 0 || ndims > CPHVB_MAXDIM)
                {
                    fprintf(stderr, "DNPY_PROC_GRID, invalid syntax or"
                                    " value at \"%s\"\n", res);
                    MPI_Abort(MPI_COMM_WORLD, -1);
                }
                tmpsizes[(ndims-1)*CPHVB_MAXDIM+(dim-1)] = dsize;
                //Go to next token.
                res = strtok(NULL, ";");
            }
        }
        //Initilization of cart_dim_strides.
        for(i=0; i<CPHVB_MAXDIM; i++)
        {
            int ndims = i+1;
            int t[CPHVB_MAXDIM];
            int d = 0;
            //Need to reverse the order to match MPI_Dims_create
            for(j=i; j>=0; j--)
                t[d++] = tmpsizes[i*CPHVB_MAXDIM+j];

            //Find a balanced distributioin of processes per direction
            //and use the restrictions specified by the user.
            MPI_Dims_create(worldsize, ndims, t);
            d = ndims;
            for(j=0; j<ndims; j++)
                msg[1+i*CPHVB_MAXDIM+j] = t[--d];
        }
        //Process grid to clients.
        msg[0] = CLUSTER_INIT_PROC_GRID;
        msg[CPHVB_MAXDIM*CPHVB_MAXDIM+1] = CLUSTER_MSG_END;
//        msg2slaves(msg, (CPHVB_MAXDIM*CPHVB_MAXDIM+2)*sizeof(cphvb_intp));
        #ifdef DEBUG
            printf("Rank 0 received msg: ");
        #endif
//        do_INIT_PROC_GRID(&msg[1]);
*/
        return 0;
    }

    while(1)
    {
        cluster_msg *msg = msg_mem;
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Bcast(msg, CLUSTER_MSG_SIZE, MPI_BYTE, 0, MPI_COMM_WORLD);

        if(msg->type == CLUSTER_SHUTDOWN)
        {
            #ifdef DEBUG
                printf("[VEM cluster] Rank %d SHUTDOWN\n", myrank);
            #endif
            break;
        }
        else if(msg->type == CLUSTER_ARRAY)
        {
            #ifdef DEBUG
                printf("[VEM cluster] Rank %d CLUSTER_ARRAY\n", myrank);
            #endif



            break;
        }
        else
        {
            assert(msg->type == CLUSTER_INST);
            assert(1==2);
        }
    }
    return 1;
}

/* Shutdown the VEM, which include a instruction flush.
 * This is a collective operation.
 *
 * @return Error codes (CPHVB_SUCCESS)
 */
cphvb_error cphvb_vem_cluster_shutdown(void)
{
    if(myrank == 0)
    {
        cluster_msg *msg = msg_mem;
    #ifdef DEBUG
        printf("[VEM cluster] Rank %d SHUTDOWN\n", myrank);
    #endif
        msg->type = CLUSTER_SHUTDOWN;
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Bcast(msg, CLUSTER_MSG_SIZE, MPI_BYTE, 0, MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return CPHVB_SUCCESS;
}


/* Create an array, which are handled by the VEM.
 *
 * @base Pointer to the base array. If NULL this is a base array
 * @type The type of data in the array
 * @ndim Number of dimentions
 * @start Index of the start element (always 0 for base-array)
 * @shape[CPHVB_MAXDIM] Number of elements in each dimention
 * @stride[CPHVB_MAXDIM] The stride for each dimention
 * @has_init_value Does the array have an initial value
 * @init_value The initial value
 * @new_array The handler for the newly created array
 * @return Error code (CPHVB_SUCCESS, CPHVB_OUT_OF_MEMORY)
 */
cphvb_error cphvb_vem_cluster_create_array(cphvb_array*   base,
                                           cphvb_type     type,
                                           cphvb_intp     ndim,
                                           cphvb_index    start,
                                           cphvb_index    shape[CPHVB_MAXDIM],
                                           cphvb_index    stride[CPHVB_MAXDIM],
                                           cphvb_intp     has_init_value,
                                           cphvb_constant init_value,
                                           cphvb_array**  new_array)
{
    cluster_msg_array *msg= msg_mem;
    cphvb_array *array    = &msg->array;
    array->owner          = CPHVB_PARENT;
    array->base           = base;
    array->type           = type;
    array->ndim           = ndim;
    array->start          = start;
    array->has_init_value = has_init_value;
    array->init_value     = init_value;
    array->data           = NULL;
    array->ref_count      = 1;
    memcpy(array->shape, shape, ndim * sizeof(cphvb_index));
    memcpy(array->stride, stride, ndim * sizeof(cphvb_index));

    msg->type = CLUSTER_ARRAY;
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(msg, CLUSTER_MSG_SIZE, MPI_BYTE, 0, MPI_COMM_WORLD);


/*
    if(array->base != NULL)
    {
        assert(array->base->base == NULL);
        assert(!has_init_value);
        ++array->base->ref_count;
        array->data = array->base->data;
    }
*/
    *new_array = array;
    return CPHVB_SUCCESS;
}


/* Check whether the instruction is supported by the VEM or not
 *
 * @return non-zero when true and zero when false
 */
cphvb_intp cphvb_vem_cluster_instruction_check(cphvb_instruction *inst)
{
    switch(inst->opcode)
    {
    case CPHVB_DESTORY:
        return 1;
    case CPHVB_RELEASE:
        return 1;
    default:
        if(ve_support.opcode[inst->opcode])
        {
            cphvb_intp i;
            cphvb_intp nop = cphvb_operands(inst->opcode);
            for(i=0; i<nop; ++i)
            {
                cphvb_type t;
                if(inst->operand[i] == CPHVB_CONSTANT)
                    t = inst->const_type[i];
                else
                    t = inst->operand[i]->type;
                if(!ve_support.type[t])
                    return 0;
            }
            return 1;
        }
        else
            return 0;
    }
}


/* Execute a list of instructions (blocking, for the time being).
 * It is required that the VEM supports all instructions in the list.
 *
 * @instruction A list of instructions to execute
 * @return Error codes (CPHVB_SUCCESS)
 */
cphvb_error cphvb_vem_cluster_execute(cphvb_intp count,
                                      cphvb_instruction inst_list[])
{
    cphvb_intp i;
    for(i=0; i<count; ++i)
    {
        cphvb_instruction *inst = &inst_list[i];
        switch(inst->opcode)
        {
        case CPHVB_DESTORY:
        {
            cphvb_array *base = cphvb_base_array(inst->operand[0]);
            if(--base->ref_count <= 0)
            {
                if(base->data != NULL)
                    free(base->data);

                if(inst->operand[0]->base != NULL)
                    free(base);
            }
            free(inst->operand[0]);
            break;
        }
        case CPHVB_RELEASE:
        {
            //Get the base
            cphvb_array *base = cphvb_base_array(inst->operand[0]);
            //Check the owner of the array
            if(base->owner != CPHVB_PARENT)
            {
                fprintf(stderr, "VEM could not perform release\n");
                exit(CPHVB_INST_ERROR);
            }
            break;
        }
        default:
        {
            cphvb_error error=1;// = cphvb_ve_simple_execute(1, inst);
            if(error)
            {
                fprintf(stderr, "cphvb_vem_execute() encountered an "
                                "error (%s) when executing %s.",
                                cphvb_error_text(error),
                                cphvb_opcode_text(inst->opcode));
                exit(error);
            }
        }
        }
    }
    return CPHVB_SUCCESS;
}


