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

#ifndef __CPHVB_H
#define __CPHVB_H

#include "cphvb_error.h"
#include "cphvb_debug.h"
#include "cphvb_opcode.h"
#include "cphvb_type.h"
#include "cphvb_array.h"
#include "cphvb_instruction.h"
#include "cphvb_bundler.h"
#include "cphvb_component.h"
#include "cphvb_compute.h"
#include "cphvb_userfunc.h"
#include "cphvb_pprint.h"
#include "cphvb_osx.h"
#include "cphvb_win.h"
#include "cphvb_memory.h"

#ifdef __cplusplus
/* C++ includes go here */
#include <cstddef>
extern "C" {
#else
/* plain C includes go here */
#include <stddef.h>
#include <stdbool.h>
#endif

/* Reduce nDarray info to a base shape
 *
 * Remove dimentions that just indicate orientation in a
 * higher dimention (Py:newaxis)
 *
 * @ndim          Number of dimentions
 * @shape[]       Number of elements in each dimention.
 * @stride[]      Stride in each dimention.
 * @base_ndim     Placeholder for base number of dimentions
 * @base_shape    Placeholder for base number of elements in each dimention.
 * @base_stride   Placeholder for base stride in each dimention.
 */
DLLEXPORT void cphvb_base_shape(cphvb_intp ndim,
                      const cphvb_index shape[],
                      const cphvb_index stride[],
                      cphvb_intp* base_ndim,
                      cphvb_index* base_shape,
                      cphvb_index* base_stride);


/* Is the data accessed continuously, and only once
 *
 * @ndim     Number of dimentions
 * @shape[]  Number of elements in each dimention.
 * @stride[] Stride in each dimention.
 * @return   Truth value indicating continuousity.
 */
DLLEXPORT bool cphvb_is_continuous(cphvb_intp ndim,
                         const cphvb_index shape[],
                         const cphvb_index stride[]);


/* Number of element in a given shape
 *
 * @ndim     Number of dimentions
 * @shape[]  Number of elements in each dimention.
 * @return   Number of element operations
 */
DLLEXPORT cphvb_index cphvb_nelements(cphvb_intp ndim,
                            const cphvb_index shape[]);


/* Calculate the offset into an array based on element index
 *
 * @ndim     Number of dimentions
 * @shape[]  Number of elements in each dimention.
 * @stride[] Stride in each dimention.
 * @element  Index of element in question
 * @return   Truth value indicating continuousity.
 */
DLLEXPORT cphvb_index cphvb_calc_offset(cphvb_intp ndim,
                              const cphvb_index shape[],
                              const cphvb_index stride[],
                              cphvb_index element);


/* Calculate the dimention boundries for shape
 *
 * @ndim      Number of dimentions
 * @shape[]   Number of elements in each dimention.
 * @dimbound  Placeholder for dimbound (return
 */
DLLEXPORT void cphvb_dimbound(cphvb_intp ndim,
                    const cphvb_index shape[],
                    cphvb_index dimbound[CPHVB_MAXDIM]);


/* Number of operands for operation
 *
 * @opcode Opcode for operation
 * @return Number of operands
 */
DLLEXPORT int cphvb_operands(cphvb_opcode opcode);

/* Number of operands in instruction
 * NB: this function handles user-defined function correctly
 * @inst Instruction
 * @return Number of operands
 */
DLLEXPORT int cphvb_operands_in_instruction(cphvb_instruction *inst);

/* Text string for operation
 *
 * @opcode Opcode for operation
 * @return Text string.
 */
DLLEXPORT const char* cphvb_opcode_text(cphvb_opcode opcode);


/* Byte size for type
 *
 * @type   Type code
 * @return Byte size
 */
DLLEXPORT int cphvb_type_size(cphvb_type type);


/* Text string for type
 *
 * @type   Type code.
 * @return Text string.
 */
DLLEXPORT const char* cphvb_type_text(cphvb_type type);


/* Text string for error code
 *
 * @error  Error code.
 * @return Text string.
 */
DLLEXPORT const char* cphvb_error_text(cphvb_error error);


/* Find the base array for a given array/view
 *
 * @view   Array/view in question
 * @return The Base array
 */
DLLEXPORT cphvb_array* cphvb_base_array(cphvb_array* view);

/* Set the data pointer for the array.
 * Can only set to non-NULL if the data ptr is already NULL
 *
 * @array The array in question
 * @data The new data pointer
 * @return Error code (CPHVB_SUCCESS, CPHVB_ERROR)
 */
DLLEXPORT cphvb_error cphvb_data_set(cphvb_array* array, cphvb_data_ptr data);

/* Get the data pointer for the array.
 *
 * @array The array in question
 * @result Output area
 * @return Error code (CPHVB_SUCCESS, CPHVB_ERROR)
 */
DLLEXPORT cphvb_error cphvb_data_get(cphvb_array* array, cphvb_data_ptr* result);

/* Allocate data memory for the given array if not already allocated.
 * If @array is a view, the data memory for the base array is allocated.
 * NB: It does NOT initiate the memory.
 * For convenience array is allowed to be NULL.
 *
 * @array  The array in question
 * @return Error code (CPHVB_SUCCESS, CPHVB_OUT_OF_MEMORY)
 */
DLLEXPORT cphvb_error cphvb_data_malloc(cphvb_array* array);

/* Frees data memory for the given array.
 * For convenience array is allowed to be NULL.
 *
 * @array  The array in question
 * @return Error code (CPHVB_SUCCESS, CPHVB_OUT_OF_MEMORY)
 */
DLLEXPORT cphvb_error cphvb_data_free(cphvb_array* array);


/* Retrive the operand type of a instruction.
 *
 * @instruction  The instruction in question
 * @operand_no Number of the operand in question
 * @return Error code (CPHVB_SUCCESS, CPHVB_OUT_OF_MEMORY)
 */
DLLEXPORT cphvb_type cphvb_type_operand(cphvb_instruction *instruction,
                              cphvb_intp operand_no);

/* Determines whether two arrays conflicts.
 *
 * @a The first array
 * @b The second array
 * @return The boolean answer
 */
DLLEXPORT bool cphvb_array_conflict(cphvb_array *a, cphvb_array *b);

/* Determines whether the array is a scalar or a broadcast view of a scalar.
 *
 * @a The array
 * @return The boolean answer
 */
DLLEXPORT bool cphvb_is_scalar(cphvb_array *array);

/* Determines whether the operand is a constant
 *
 * @o The operand
 * @return The boolean answer
 */
DLLEXPORT bool cphvb_is_constant(cphvb_array* o);


#ifdef __cplusplus
}
#endif

#endif
