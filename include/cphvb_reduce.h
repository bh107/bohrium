/*
 * Copyright 2011 Troels Blum <troels@blum.dk>
 *
 * This file is part of cphVB <http://code.google.com/p/cphvb/>.
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

#ifndef __CPHVB_REDUCE_H
#define __CPHVB_REDUCE_H

/* This header file is for the user-defined reduce function.
 * We include an implementation with the default cphVB project.
 * However, it is possible to overwrite the default implementation
 * by specifying an alternative library in the config.ini.
 */


#ifdef __cplusplus
extern "C" {
#endif

//The type of the user-defined reduce function.
typedef struct
{
    //User-defined function header with two operands.
    CPHVB_USER_FUNC_HEADER(2)
    //The Axis to reduce
    cphvb_error   axis;
    //The opcode to reduce with
    cphvb_opcode  opcode;
} cphvb_reduce_type;


#ifdef __cplusplus
}
#endif

#endif
