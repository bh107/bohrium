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
 * along with cphVB.  If not, see <http://www.gnu.org/licenses/>.
 */
#include <assert.h>
#include <string.h>

#include <cphvb_vem.h>
#include <cphvb.h>
#include "private.h"

//UID count.
static cphvb_int32 uid_count=0;


cphvb_error cphvb_vem_init(void)
{

    return CPHVB_SUCCESS;
}

cphvb_error cphvb_vem_create_array(cphvb_array*   base,
                                   cphvb_type     type,
                                   cphvb_int32    ndim,
                                   cphvb_index    start,
                                   cphvb_index    shape[CPHVB_MAXDIM],
                                   cphvb_index    stride[CPHVB_MAXDIM],
                                   cphvb_bool     has_init_value,
                                   cphvb_constant init_value,
                                   cphvb_int32    *uid)
{
    cphvb_array *array    = malloc(sizeof(cphvb_array));
    if(array == NULL)
        return CPHVB_OUT_OF_MEMORY;

    array->owner          = CPHVB_BRIDGE;
    array->base           = base;
    array->type           = type;
    array->ndim           = ndim;
    array->start          = start;
    array->has_init_value = has_init_value;
    array->init_value     = init_value;
    array->ref_count      = 1;
    memcpy(array->shape, shape, ndim * sizeof(cphvb_index));
    memcpy(array->stride, stride, ndim * sizeof(cphvb_index));

    if(array->base != NULL)
    {
        assert(!has_init_value);
        ++array->base->ref_count;
        array->data = array->base->data;
    }
    *uid = ++uid_count;
    return CPHVB_SUCCESS;
}


cphvb_error cphvb_vem_execute(cphvb_int32 instruction_count,
                              char* instruction_list);

cphvb_error cphvb_vem_simple_shutdown(void);


