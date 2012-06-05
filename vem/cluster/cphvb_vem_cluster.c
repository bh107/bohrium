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
#include <math.h>
#include <assert.h>
#include "cphvb_vem_cluster.h"
#include "process_grid.h"

//The VEM components
static cphvb_com **coms;

//Our self
static cphvb_com *myself;

//Function pointers to the NODE-VEM.
static cphvb_init vem_init;
static cphvb_execute vem_execute;
static cphvb_shutdown vem_shutdown;
static cphvb_reg_func vem_reg_func;
static cphvb_create_array vem_create_array;

//Number of user-defined functions registered.
static cphvb_intp userfunc_count = 0;

cphvb_error cphvb_vem_cluster_init(cphvb_com *self)
{
    cphvb_intp children_count;
    cphvb_error err;
    myself = self;

    //Initiate the process grid (incl. MPI)
    pgrid_init();

    cphvb_com_children(self, &children_count, &coms);
    vem_init = coms[0]->init;
    vem_execute = coms[0]->execute;
    vem_shutdown = coms[0]->shutdown;
    vem_reg_func = coms[0]->reg_func;
    vem_create_array = coms[0]->create_array;

    //Let us initiate the simple VE and register what it supports.
    err = vem_init(coms[0]);
    if(err)
        return err;

    return CPHVB_SUCCESS;
}

cphvb_error cphvb_vem_cluster_execute(cphvb_intp instruction_count,
                                      cphvb_instruction* instruction_list)
{
    cphvb_intp i;
    for(i=0; i<instruction_count; ++i)
    {
        cphvb_instruction* inst = &instruction_list[i];
        printf("cphvb_vem_cluster_execute: %s\n", cphvb_opcode_text(inst->opcode));


        switch(inst->opcode)
        {
        case CPHVB_DESTROY:
        {
            cphvb_array* base = cphvb_base_array(inst->operand[0]);
            --base->ref_count; //decrease refcount
            assert(inst->operand[0]->base == NULL);
            if(base->ref_count <= 0)
            {
                dndarray *tmp = (dndarray*)inst->operand[0];
                inst->operand[0] = tmp->child_ary;
                cphvb_error e = vem_execute(1, inst);
                if(e != CPHVB_SUCCESS)
                    return e;
                if(tmp->data != NULL)
                    free(tmp->data);
                free(tmp);
            }

            break;
        }
/*
        case CPHVB_SYNC:
        {
            cphvb_array* base = cphvb_base_array(inst->operand[0]);
            switch (base->owner)
            {
            case CPHVB_PARENT:
            case CPHVB_SELF:
                //The owner is not down stream so we do nothing
                inst->opcode = CPHVB_NONE;
                --valid_instruction_count;
                break;
            default:
                //The owner is downsteam so send the sync down
                //and take ownership
                inst->operand[0] = base;
                arrayManager->changeOwnerPending(base,CPHVB_SELF);
            }
            break;
        }
        case CPHVB_USERFUNC:
        {
            cphvb_userfunc *uf = inst->userfunc;
            //The children should own the output arrays.
            for(int i = 0; i < uf->nout; ++i)
            {
                cphvb_array* base = cphvb_base_array(uf->operand[i]);
                base->owner = CPHVB_CHILD;
            }
            //We should own the input arrays.
            for(int i = uf->nout; i < uf->nout + uf->nin; ++i)
            {
                cphvb_array* base = cphvb_base_array(uf->operand[i]);
                if(base->owner == CPHVB_PARENT)
                {
                    base->owner = CPHVB_SELF;
                }
            }
            break;
        }
*/
        default:
        {
/*
            cphvb_array* base = cphvb_base_array(inst->operand[0]);
            // "Regular" operation: set ownership and send down stream
            base->owner = CPHVB_CHILD;//The child owns the output ary.
            for (int i = 1; i < cphvb_operands(inst->opcode); ++i)
            {
                if(cphvb_base_array(inst->operand[i])->owner == CPHVB_PARENT)
                {
                    cphvb_base_array(inst->operand[i])->owner = CPHVB_SELF;
                }
            }
*/
        }
        }
    }

    return CPHVB_SUCCESS;
}

cphvb_error cphvb_vem_cluster_shutdown(void)
{
    cphvb_error err;
    pgrid_finalize();
    err = vem_shutdown();
    cphvb_com_free(coms[0]);//Only got one child.
    free(coms);
    return err;
}

/* Registre a new user-defined function.
 *
 * @fun Name of the function e.g. myfunc
 * @id Identifier for the new function. The bridge should set the
 *     initial value to Zero. (in/out-put)
 * @return Error codes (CPHVB_SUCCESS)
 */
cphvb_error cphvb_vem_cluster_reg_func(char *fun, cphvb_intp *id)
{
    if(*id == 0)//Only if parent didn't set the ID.
        *id = ++userfunc_count;

    return vem_reg_func(fun, id);
}



/* Create an array, which are handled by the VEM.
 *
 * @base Pointer to the base array. If NULL this is a base array
 * @type The type of data in the array
 * @ndim Number of dimentions
 * @start Index of the start element (always 0 for base-array)
 * @shape[CPHVB_MAXDIM] Number of elements in each dimention
 * @stride[CPHVB_MAXDIM] The stride for each dimention
 * @new_array The handler for the newly created array
 * @return Error code (CPHVB_SUCCESS, CPHVB_OUT_OF_MEMORY)
 */
cphvb_error cphvb_vem_cluster_create_array(cphvb_array*   base,
                                           cphvb_type     type,
                                           cphvb_intp     ndim,
                                           cphvb_index    start,
                                           cphvb_index    shape[CPHVB_MAXDIM],
                                           cphvb_index    stride[CPHVB_MAXDIM],
                                           cphvb_array**  new_array)
{
    printf("cphvb_vem_cluster_create_array\n");

    if(base == NULL)
    {
        int *cdims = pgrid_dim_size(ndim);
        cphvb_intp i;
        int cartcoord[CPHVB_MAXDIM];
        dndarray *ary = malloc(sizeof(dndarray));

        if(ary == NULL)
            return CPHVB_OUT_OF_MEMORY;

        ary->base           = base;
        ary->type           = type;
        ary->ndim           = ndim;
        ary->start          = start;
        ary->data           = NULL;
        ary->ref_count      = 1;
        memcpy(ary->shape, shape, ndim * sizeof(cphvb_index));
        memcpy(ary->stride, stride, ndim * sizeof(cphvb_index));

        //Get process grid coords.
        rank2pgrid(ndim, myrank, cartcoord);

        //Accumulate the total number of local sizes and save it.
        cphvb_intp localsize = 1;
        ary->nblocks = 1;
        for(i=0; i < ndim; i++)
        {
            ary->localdims[i] = pgrid_numroc(ary->shape[i], blocksize, cartcoord[i], cdims[i], 0);

            localsize *= ary->localdims[i];
            ary->localblockdims[i] = ceil(ary->localdims[i] / (double) blocksize);
            ary->blockdims[i] = ceil(ary->shape[i] / (double) blocksize);
            ary->nblocks *= ary->blockdims[i];
        }
        ary->localsize = localsize;
        if(ary->localsize == 0)
        {
            memset(ary->localdims, 0, ary->ndim * sizeof(cphvb_intp));
            memset(ary->localblockdims, 0, ary->ndim * sizeof(cphvb_intp));
        }
        if(ary->nblocks == 0)
            memset(ary->blockdims, 0, ary->ndim * sizeof(cphvb_intp));

        //Compute local stride stride.
        cphvb_intp s = 1;
        for(i=ndim; i >= 0; --i)
        {
            ary->localstride[i] = s;
            s *= ary->localdims[i];
        }

        vem_create_array(base, type, ndim, start, ary->localdims, ary->localstride, &ary->child_ary);

        *new_array = (cphvb_array*) ary;
    }

    return CPHVB_SUCCESS;
}
