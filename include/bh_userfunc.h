/*
This file is part of Bohrium and copyright (c) 2012 the Bohrium
team <http://www.bh107.org>.

Bohrium is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as 
published by the Free Software Foundation, either version 3 
of the License, or (at your option) any later version.

Bohrium is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the 
GNU Lesser General Public License along with Bohrium. 

If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __CPHVB_USERFUNC_H
#define __CPHVB_USERFUNC_H

/* This header file is for the user-defined functions.
 * We include an implementation with the default Bohrium project.
 * However, it is possible to overwrite the default implementation
 * by specifying an alternative library in the config.ini.
 */

#include "bh_instruction.h"
#include "bh_win.h"

#ifdef __cplusplus
extern "C" {
#endif

//The type of the user-defined reduce function.
typedef struct
{
    //User-defined function header with two operands.
    CPHVB_USER_FUNC_HEADER(2)
    //The Axis to reduce
    bh_index   axis;
    //The opcode to reduce with
    bh_opcode  opcode;
} bh_reduce_type;

DLLEXPORT bh_error bh_reduce(bh_userfunc* arg, void* ve_arg);

//The type of the user-defined aggregate function.
typedef struct
{
    //User-defined function header with two operands.
    CPHVB_USER_FUNC_HEADER(2)
    //The opcode to aggregate with
    bh_opcode  opcode;
} bh_aggregate_type;

DLLEXPORT bh_error bh_aggregate(bh_userfunc* arg, void* ve_arg);

//The type of the user-defined random function.
typedef struct
{
    //User-defined function header with one operands.
    CPHVB_USER_FUNC_HEADER(1)
} bh_random_type;

DLLEXPORT bh_error bh_random(bh_userfunc* arg, void* ve_arg);

//The type of the user-defined matrix multiplication function.
typedef struct
{
    //User-defined function header with three operands.
    CPHVB_USER_FUNC_HEADER(3)
} bh_matmul_type;

DLLEXPORT bh_error bh_matmul(bh_userfunc* arg, void* ve_arg);

//The type of the user-defined lu factorization function.
typedef struct
{
    //User-defined function header with two operands.
    CPHVB_USER_FUNC_HEADER(2)
} bh_lu_type;

DLLEXPORT bh_error bh_lu(bh_userfunc* arg, void* ve_arg);

//The type of the user-defined fft function.
typedef struct
{
    //User-defined function header with two operands.
    CPHVB_USER_FUNC_HEADER(2)
} bh_fft_type;

DLLEXPORT bh_error bh_fft(bh_userfunc* arg, void* ve_arg);
DLLEXPORT bh_error bh_fft2(bh_userfunc* arg, void* ve_arg);

//The type of the user-defined nselect (maxn, minn, etc.) function.
typedef struct
{
    //User-defined function header with three operands 
    //(one input and two outputs).
    CPHVB_USER_FUNC_HEADER(3)
    //The 'n' in n-select.
    bh_intp   n;
    //The axis to n-select.
    bh_intp   axis;
    //The opcode to use with n-select.
    bh_opcode  opcode;
} bh_nselect_type;

DLLEXPORT bh_error bh_nselect(bh_userfunc* arg, void* ve_arg);

#ifdef __cplusplus
}
#endif

#endif
