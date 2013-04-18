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

#ifndef __BH_H
#define __BH_H

#include "bh_error.h"
#include "bh_debug.h"
#include "bh_opcode.h"
#include "bh_type.h"
#include "bh_array.h"
#include "bh_instruction.h"
#include "bh_bundler.h"
#include "bh_component.h"
#include "bh_compute.h"
#include "bh_userfunc.h"
#include "bh_pprint.h"
#include "bh_osx.h"
#include "bh_win.h"
#include "bh_memory.h"
#include "bh_timing.h"

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
DLLEXPORT void bh_base_shape(bh_intp ndim,
                      const bh_index shape[],
                      const bh_index stride[],
                      bh_intp* base_ndim,
                      bh_index* base_shape,
                      bh_index* base_stride);


/* Is the data accessed continuously, and only once
 *
 * @ndim     Number of dimentions
 * @shape[]  Number of elements in each dimention.
 * @stride[] Stride in each dimention.
 * @return   Truth value indicating continuousity.
 */
DLLEXPORT bool bh_is_continuous(bh_intp ndim,
                         const bh_index shape[],
                         const bh_index stride[]);


/* Number of element in a given shape
 *
 * @ndim     Number of dimentions
 * @shape[]  Number of elements in each dimention.
 * @return   Number of element operations
 */
DLLEXPORT bh_index bh_nelements(bh_intp ndim,
                            const bh_index shape[]);


/* Size of the array data
 *
 * @array    The array in question
 * @return   The size of the array data in bytes
 */
DLLEXPORT bh_index bh_array_size(const bh_array *array);


/* Calculate the offset into an array based on element index
 *
 * @ndim     Number of dimentions
 * @shape[]  Number of elements in each dimention.
 * @stride[] Stride in each dimention.
 * @element  Index of element in question
 * @return   Truth value indicating continuousity.
 */
DLLEXPORT bh_index bh_calc_offset(bh_intp ndim,
                              const bh_index shape[],
                              const bh_index stride[],
                              bh_index element);


/* Calculate the dimention boundries for shape
 *
 * @ndim      Number of dimentions
 * @shape[]   Number of elements in each dimention.
 * @dimbound  Placeholder for dimbound (return
 */
DLLEXPORT void bh_dimbound(bh_intp ndim,
                    const bh_index shape[],
                    bh_index dimbound[BH_MAXDIM]);


/* Set the array stride to contiguous row-major
 *
 * @array    The array in question
 * @return   The total number of elements in array
 */
bh_intp bh_set_contiguous_stride(bh_array *array);


/* Number of operands for operation
 *
 * @opcode Opcode for operation
 * @return Number of operands
 */
DLLEXPORT int bh_operands(bh_opcode opcode);

/* Number of operands in instruction
 * NB: this function handles user-defined function correctly
 * @inst Instruction
 * @return Number of operands
 */
DLLEXPORT int bh_operands_in_instruction(const bh_instruction *inst);

/* Text string for operation
 *
 * @opcode Opcode for operation
 * @return Text string.
 */
DLLEXPORT const char* bh_opcode_text(bh_opcode opcode);

/* Determines if the operation is a system operation
 *
 * @opcode Opcode for operation
 * @return TRUE if the operation is a system opcode, FALSE otherwise
 */
DLLEXPORT bool bh_opcode_is_system(bh_opcode opcode);

/* Determines if the operation is performed elementwise
 *
 * @opcode Opcode for operation
 * @return TRUE if the operation is performed elementwise, FALSE otherwise
 */
DLLEXPORT bool bh_opcode_is_elementwise(bh_opcode opcode);


/* Byte size for type
 *
 * @type   Type code
 * @return Byte size
 */
DLLEXPORT int bh_type_size(bh_type type);


/* Text string for type
 *
 * @type   Type code.
 * @return Text string.
 */
DLLEXPORT const char* bh_type_text(bh_type type);


/* Text string for error code
 *
 * @error  Error code.
 * @return Text string.
 */
DLLEXPORT const char* bh_error_text(bh_error error);


/* Find the base array for a given array/view
 *
 * @view   Array/view in question
 * @return The Base array
 */
DLLEXPORT bh_array* bh_base_array(const bh_array* view);

/* Set the data pointer for the array.
 * Can only set to non-NULL if the data ptr is already NULL
 *
 * @array The array in question
 * @data The new data pointer
 * @return Error code (BH_SUCCESS, BH_ERROR)
 */
DLLEXPORT bh_error bh_data_set(bh_array* array, bh_data_ptr data);

/* Get the data pointer for the array.
 *
 * @array The array in question
 * @result Output area
 * @return Error code (BH_SUCCESS, BH_ERROR)
 */
DLLEXPORT bh_error bh_data_get(bh_array* array, bh_data_ptr* result);

/* Allocate data memory for the given array if not already allocated.
 * If @array is a view, the data memory for the base array is allocated.
 * NB: It does NOT initiate the memory.
 * For convenience array is allowed to be NULL.
 *
 * @array  The array in question
 * @return Error code (BH_SUCCESS, BH_OUT_OF_MEMORY)
 */
DLLEXPORT bh_error bh_data_malloc(bh_array* array);

/* Frees data memory for the given array.
 * For convenience array is allowed to be NULL.
 *
 * @array  The array in question
 * @return Error code (BH_SUCCESS, BH_OUT_OF_MEMORY)
 */
DLLEXPORT bh_error bh_data_free(bh_array* array);

/* Retrive the operands of a instruction.
 *
 * @instruction  The instruction in question
 * @return The operand list
 */
DLLEXPORT bh_array **bh_inst_operands(const bh_instruction *instruction);

/* Retrive the operand type of a instruction.
 *
 * @instruction  The instruction in question
 * @operand_no Number of the operand in question
 * @return The operand type
 */
DLLEXPORT bh_type bh_type_operand(const bh_instruction *instruction,
                                        bh_intp operand_no);

/* Determines whether two arrays overlap.
 * NB: This functions may return True on non-overlapping arrays. 
 *     But will always return False on overlapping arrays.
 * 
 * @a The first array
 * @b The second array
 * @return The boolean answer
 */
bool bh_array_overlap(const bh_array *a, const bh_array *b);


/* Determines whether the array is a scalar or a broadcast view of a scalar.
 *
 * @a The array
 * @return The boolean answer
 */
DLLEXPORT bool bh_is_scalar(const bh_array *array);

/* Determines whether the operand is a constant
 *
 * @o The operand
 * @return The boolean answer
 */
DLLEXPORT bool bh_is_constant(const bh_array* o);

/* Determines whether the two views are the same
 *
 * @a The first array
 * @b The second array
 * @return The boolean answer
 */
DLLEXPORT bool bh_same_view(const bh_array* a, const bh_array* b);

/* Determines whether two array(views)s access some of the same data points
 *
 * @a The first array
 * @b The second array
 * @return The boolean answer
 */
DLLEXPORT bool bh_disjoint_views(const bh_array *a, const bh_array *b);


#ifdef __cplusplus
}
#endif

#endif

