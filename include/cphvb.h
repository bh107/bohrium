/*
 * Copyright 2011 Troels Blum <troels@blum.dk>
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

#ifndef __CPHVB_H
#define __CPHVB_H

#include "cphvb_error.h"
#include "cphvb_opcode.h"
#include "cphvb_type.h"
#include "cphvb_array.h"
#include "cphvb_instruction.h"

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
void cphvb_base_shape(cphvb_int32 ndim,
                      const cphvb_index shape[],
                      const cphvb_index stride[],
                      cphvb_int32* base_ndim,
                      cphvb_index* base_shape,
                      cphvb_index* base_stride);


/* Is the data accessed continuously, and only once
 *
 * @ndim     Number of dimentions
 * @shape[]  Number of elements in each dimention.
 * @stride[] Stride in each dimention.
 * @return   Truth value indicating continuousity.
 */
bool cphvb_is_continuous(cphvb_int32 ndim,
                         const cphvb_index shape[],
                         const cphvb_index stride[]);


/* Number of element in a given shape
 *
 * @ndim     Number of dimentions
 * @shape[]  Number of elements in each dimention.
 * @return   Number of element operations
 */
size_t cphvb_nelements(cphvb_int32 ndim,
                       const cphvb_index shape[]);


/* Calculate the offset into an array based on element index
 *
 * @ndim     Number of dimentions
 * @shape[]  Number of elements in each dimention.
 * @stride[] Stride in each dimention.
 * @element  Index of element in question
 * @return   Truth value indicating continuousity.
 */
cphvb_index cphvb_calc_offset(cphvb_int32 ndim,
                              const cphvb_index shape[],
                              const cphvb_index stride[],
                              cphvb_index element);


/* Pretty print instruction
 *
 * Mainly for debugging purposes.
 *
 * @inst   Instruction to print.
 * @size   Write at most this many bytes to buffer.
 * @buf    Buffer to contain the string.
 * @return Number of characters printed.
 */
int cphvb_snprint(const cphvb_instruction* inst,
                  size_t size,
                  char* buf);


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


#ifdef __cplusplus
}
#endif

#endif
