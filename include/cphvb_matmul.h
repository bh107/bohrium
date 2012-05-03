/*
 * Copyright 2011 Kenneth Skovhede <kenneth@hexad.dk>
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

#ifndef __CPHVB_MATMUL_H
#define __CPHVB_MATMUL_H

/* This header file is for the user-defined matmul function.
 * We include an implementation with the default cphVB project.
 * However, it is possible to overwrite the default implementation
 * by specifying an alternative library in the config.ini.
 */

#include "cphvb_instruction.h"
#include "cphvb_win.h"

#ifdef __cplusplus
extern "C" {
#endif

//The type of the user-defined reduce function.
typedef struct
{
    //User-defined function header with three operands.
    CPHVB_USER_FUNC_HEADER(3)
} cphvb_matmul_type;

DLLEXPORT cphvb_error cphvb_matmul(cphvb_userfunc* arg, void* ve_arg);

#ifdef __cplusplus
}
#endif

#endif
