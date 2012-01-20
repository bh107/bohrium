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

static cphvb_com *myself;

cphvb_error cphvb_vem_cluster_init(cphvb_com *self)
{
    myself = self;
    return CPHVB_SUCCESS;
}

cphvb_error cphvb_vem_cluster_execute(cphvb_intp instruction_count,
                                      cphvb_instruction* instruction_list)
{
    cphvb_error res = CPHVB_SUCCESS;


    return res;

}

cphvb_error cphvb_vem_cluster_shutdown( void ) {

    return CPHVB_SUCCESS;

}

cphvb_error cphvb_vem_cluster_reg_func(char *lib, char *fun, cphvb_intp *id)
{
    return CPHVB_SUCCESS;
}

