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
#include "bh_pprint.h"
#include "bh_osx.h"
#include "bh_win.h"
#include "bh_memory.h"
#include "bh_ir.h"
#include "bh_ir_map.h"
#include "bh_signal.h"

#ifdef __cplusplus
/* C++ includes go here */
#include <cstddef>
extern "C" {
#else
/* plain C includes go here */
#include <stddef.h>
#include <stdbool.h>
#endif

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) >= (b) ? (a) : (b))

/* Find the base array for a given view
 *
 * @view   The view in question
 * @return The Base array
 */
#define bh_base_array(view) ((view)->base)

/* Number of non-broadcasted elements in a given view
 *
 * @view    The view in question.
 * @return  Number of elements.
 */
bh_index bh_nelements_nbcast(const bh_view *view);

/* Number of element in a given shape
 *
 * @ndim     Number of dimentions
 * @shape[]  Number of elements in each dimention.
 * @return   Number of element operations
 */
DLLEXPORT bh_index bh_nelements(bh_intp ndim,
                            const bh_index shape[]);

/* Size of the base array in bytes
 *
 * @base    The base in question
 * @return  The size of the base array in bytes
 */
bh_index bh_base_size(const bh_base *base);

/* Set the view stride to contiguous row-major
 *
 * @view    The view in question
 * @return  The total number of elements in view
 */
DLLEXPORT bh_intp bh_set_contiguous_stride(bh_view *view);

/* Updates the view with the complete base
 *
 * @view    The view to update (in-/out-put)
 * @base    The base assign to the view
 * @return  The total number of elements in view
 */
DLLEXPORT void bh_assign_complete_base(bh_view *view, bh_base *base);

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

/* Determines if the types are acceptable for the operation
 *
 * @opcode Opcode for operation
 * @outtype The type of the output
 * @inputtype1 The type of the first input operand
 * @inputtype2 The type of the second input operand
 * @constanttype The type of the constant
 * @return TRUE if the types are accepted, FALSE otherwise
 */
DLLEXPORT bool bh_validate_types(bh_opcode opcode, bh_type outtype, bh_type inputtype1, bh_type inputtype2, bh_type constanttype);

/* Determines if the types are acceptable for the operation,
 * and provides a suggestion for converting them
 *
 * @opcode Opcode for operation
 * @outtype The type of the output
 * @inputtype1 The type of the first input operand
 * @inputtype2 The type of the second input operand
 * @constanttype The type of the constant
 * @return TRUE if the types can be converted, FALSE otherwise
 */
DLLEXPORT bool bh_get_type_conversion(bh_opcode opcode, bh_type outtype, bh_type* inputtype1, bh_type* inputtype2, bh_type* constanttype);

/**
 *  Deduct a numerical value representing the types of the given instruction.
 *
 *  @param instr The instruction for which to deduct a signature.
 *  @return The deducted signature.
 */
DLLEXPORT int bh_type_sig(bh_instruction *instr);

/**
 *  Determine whether the given typesig, in the coding produced by bh_typesig, is valid.
 *
 *  @param instr The instruction for which to deduct a signature.
 *  @return The deducted signature.
 */
DLLEXPORT bool bh_type_sig_check(int type_sig);

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

/* Set the data pointer for the view.
 * Can only set to non-NULL if the data ptr is already NULL
 *
 * @view   The view in question
 * @data   The new data pointer
 * @return Error code (BH_SUCCESS, BH_ERROR)
 */
DLLEXPORT bh_error bh_data_set(bh_view* view, bh_data_ptr data);

/* Get the data pointer for the view.
 *
 * @view    The view in question
 * @result  Output data pointer
 * @return  Error code (BH_SUCCESS, BH_ERROR)
 */
DLLEXPORT bh_error bh_data_get(bh_view* view, bh_data_ptr* result);

/* Allocate data memory for the given base if not already allocated.
 * For convenience, the base is allowed to be NULL.
 *
 * @base    The base in question
 * @return  Error code (BH_SUCCESS, BH_ERROR, BH_OUT_OF_MEMORY)
 */
DLLEXPORT bh_error bh_data_malloc(bh_base* base);

/* Frees data memory for the given view.
 * For convenience, the view is allowed to be NULL.
 *
 * @base    The base in question
 * @return  Error code (BH_SUCCESS, BH_ERROR)
 */
bh_error bh_data_free(bh_base* base);

/* Retrive the operands of a instruction.
 *
 * @instruction  The instruction in question
 * @return The operand list
 */
DLLEXPORT bh_view *bh_inst_operands(bh_instruction *instruction);

/* Determines whether the view is a scalar or a broadcasted scalar.
 *
 * @view The view
 * @return The boolean answer
 */
DLLEXPORT bool bh_is_scalar(const bh_view* view);

/* Determines whether the operand is a constant
 *
 * @o The operand
 * @return The boolean answer
 */
DLLEXPORT bool bh_is_constant(const bh_view* o);

/* Flag operand as a constant
 *
 * @o      The operand
 */
DLLEXPORT void bh_flag_constant(bh_view* o);

/* Determines whether two views access some of the same data points
 * NB: This functions may return True on non-overlapping views.
 *     But will always return False on overlapping views.
 *
 * @a The first view
 * @b The second view
 * @return The boolean answer
 */
DLLEXPORT bool bh_view_disjoint(const bh_view *a, const bh_view *b);

/* Determines whether two views are identical and points
 * to the same base array.
 *
 * @a The first view
 * @b The second view
 * @return The boolean answer
 */
DLLEXPORT bool bh_view_identical(const bh_view *a, const bh_view *b);

/* Determines whether two views are aligned and points
 * to the same base array.
 *
 * @a The first view
 * @b The second view
 * @return The boolean answer
 */
DLLEXPORT bool bh_view_aligned(const bh_view *a, const bh_view *b);

#ifdef __cplusplus
}
#endif

#endif

