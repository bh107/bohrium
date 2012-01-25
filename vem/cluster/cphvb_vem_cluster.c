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
    cphvb_error res = CPHVB_SUCCESS;


    return res;

}

cphvb_error cphvb_vem_cluster_shutdown(void)
{
    pgrid_finalize();
    return CPHVB_SUCCESS;
}

/* Registre a new user-defined function.
 *
 * @lib Name of the shared library e.g. libmyfunc.so
 *      When NULL the default library is used.
 * @fun Name of the function e.g. myfunc
 * @id Identifier for the new function. The bridge should set the
 *     initial value to Zero. (in/out-put)
 * @return Error codes (CPHVB_SUCCESS)
 */
cphvb_error cphvb_vem_cluster_reg_func(char *lib, char *fun, cphvb_intp *id)
{
    if(*id == 0)//Only if parent didn't set the ID.
        *id = ++userfunc_count;

    return vem_reg_func(lib, fun, id);
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

    printf("cphvb_vem_cluster_create_array\n");

    //We will only tell the NODE-VEM about base arrays at this time.
    if(base == NULL)
    {

        vem_create_array(base, type, ndim, start, shape, stride, has_init_value, init_value, new_array);
    }

    return CPHVB_SUCCESS;
}
