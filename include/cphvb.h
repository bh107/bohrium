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

#ifndef __CPHVB_H
#define __CPHVB_H

#include "cphvb_error.h"
#include "cphvb_opcode.h"
#include "cphvb_type.h"
#include "cphvb_array.h"
#include "cphvb_instruction.h"
#include "cphvb_component.h"

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
void cphvb_base_shape(cphvb_intp ndim,
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
bool cphvb_is_continuous(cphvb_intp ndim,
                         const cphvb_index shape[],
                         const cphvb_index stride[]);


/* Number of element in a given shape
 *
 * @ndim     Number of dimentions
 * @shape[]  Number of elements in each dimention.
 * @return   Number of element operations
 */
cphvb_index cphvb_nelements(cphvb_intp ndim,
                            const cphvb_index shape[]);


/* Calculate the offset into an array based on element index
 *
 * @ndim     Number of dimentions
 * @shape[]  Number of elements in each dimention.
 * @stride[] Stride in each dimention.
 * @element  Index of element in question
 * @return   Truth value indicating continuousity.
 */
cphvb_index cphvb_calc_offset(cphvb_intp ndim,
                              const cphvb_index shape[],
                              const cphvb_index stride[],
                              cphvb_index element);


/* Calculate the dimention boundries for shape
 *
 * @ndim      Number of dimentions
 * @shape[]   Number of elements in each dimention.
 * @dimbound  Placeholder for dimbound (return
 */
void cphvb_dimbound(cphvb_intp ndim,
                    const cphvb_index shape[],
                    cphvb_index dimbound[CPHVB_MAXDIM]);


/* Number of operands for operation
 *
 * @opcode Opcode for operation
 * @return Number of operands
 */
int cphvb_operands(cphvb_opcode opcode);


/* Text string for operation
 *
 * @opcode Opcode for operation
 * @return Text string.
 */
const char* cphvb_opcode_text(cphvb_opcode opcode);


/* Byte size for type
 *
 * @type   Type code
 * @return Byte size
 */
int cphvb_type_size(cphvb_type type);


/* Text string for type
 *
 * @type   Type code.
 * @return Text string.
 */
const char* cphvb_type_text(cphvb_type type);


/* Text string for error code
 *
 * @error  Error code.
 * @return Text string.
 */
const char* cphvb_error_text(cphvb_error error);


/* Find the base array for a given array/view
 *
 * @view   Array/view in question
 * @return The Base array
 */
cphvb_array* cphvb_base_array(cphvb_array* view);


/* Allocate data memory for the given array
 * Initialize the memory if needed
 *
 * @array  The array in question
 * @return Error code (CPHVB_SUCCESS, CPHVB_OUT_OF_MEMORY)
 */
cphvb_error cphvb_malloc_array_data(cphvb_array* array);


/* Retrive the operand type of a instruction.
 *
 * @instruction  The instruction in question
 * @operand_no Number of the operand in question
 * @return Error code (CPHVB_SUCCESS, CPHVB_OUT_OF_MEMORY)
 */
cphvb_type cphvb_type_operand(cphvb_instruction *instruction,
                              cphvb_intp operand_no);

#ifdef __cplusplus
}
#endif

#endif
