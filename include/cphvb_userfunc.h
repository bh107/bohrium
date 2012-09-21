/*
This file is part of cphVB and copyright (c) 2012 the cphVB team:
http://cphvb.bitbucket.org

cphVB is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as 
published by the Free Software Foundation, either version 3 
of the License, or (at your option) any later version.

cphVB is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the 
GNU Lesser General Public License along with cphVB. 

If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __CPHVB_USERFUNC_H
#define __CPHVB_USERFUNC_H

/* This header file is for the user-defined functions.
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
    //User-defined function header with two operands.
    CPHVB_USER_FUNC_HEADER(2)
    //The Axis to reduce
    cphvb_index   axis;
    //The opcode to reduce with
    cphvb_opcode  opcode;
} cphvb_reduce_type;

DLLEXPORT cphvb_error cphvb_reduce(cphvb_userfunc* arg, void* ve_arg);

//The type of the user-defined random function.
typedef struct
{
    //User-defined function header with one operands.
    CPHVB_USER_FUNC_HEADER(1)
} cphvb_random_type;

DLLEXPORT cphvb_error cphvb_random(cphvb_userfunc* arg, void* ve_arg);

//The type of the user-defined matrix multiplication function.
typedef struct
{
    //User-defined function header with three operands.
    CPHVB_USER_FUNC_HEADER(3)
} cphvb_matmul_type;

DLLEXPORT cphvb_error cphvb_matmul(cphvb_userfunc* arg, void* ve_arg);

//The type of the user-defined lu factorization function.
typedef struct
{
    //User-defined function header with two operands.
    CPHVB_USER_FUNC_HEADER(2)
} cphvb_lu_type;

DLLEXPORT cphvb_error cphvb_lu(cphvb_userfunc* arg, void* ve_arg);

//The type of the user-defined fft function.
typedef struct
{
    //User-defined function header with two operands.
    CPHVB_USER_FUNC_HEADER(2)
} cphvb_fft_type;

DLLEXPORT cphvb_error cphvb_fft(cphvb_userfunc* arg, void* ve_arg);
DLLEXPORT cphvb_error cphvb_fft2(cphvb_userfunc* arg, void* ve_arg);

//The type of the user-defined nselect (maxn, minn, etc.) function.
typedef struct
{
    //User-defined function header with three operands 
    //(one input and two outputs).
    CPHVB_USER_FUNC_HEADER(3)
    //The 'n' in n-select.
    cphvb_intp   n;
    //The axis to n-select.
    cphvb_intp   axis;
    //The opcode to use with n-select.
    cphvb_opcode  opcode;
} cphvb_nselect_type;

DLLEXPORT cphvb_error cphvb_nselect(cphvb_userfunc* arg, void* ve_arg);

#ifdef __cplusplus
}
#endif

#endif
