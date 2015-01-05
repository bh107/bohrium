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
#ifndef __BH_PPRINT_H
#define __BH_PPRINT_H

#include "bh_opcode.h"
#include "bh_array.h"
#include "bh_error.h"
#include "bh_ir.h"
#include "bh_type.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Pretty print an base.
 *
 * @op      The base in question
 * @buf     Output buffer (must have sufficient size)
 */
DLLEXPORT void bh_sprint_base(const bh_base *base, char buf[]);

/* Pretty print an array.
 *
 * @view  The array view in question
 */
DLLEXPORT void bh_pprint_array(const bh_view *view);

/* Pretty print an view.
 *
 * @op      The view in question
 * @buf     Output buffer (must have sufficient size)
 */
DLLEXPORT void bh_sprint_view(const bh_view *op, char buf[]);

/* Pretty print an instruction.
 *
 * @instr  The instruction in question
 */
DLLEXPORT void bh_pprint_instr(const bh_instruction *instr);

/* Pretty print an instruction.
 *
 * @instr   The instruction in question
 * @buf     Output buffer (must have sufficient size)
 * @newline The new line string
 */
DLLEXPORT void bh_sprint_instr(const bh_instruction *instr, char buf[],
                               const char newline[]);

/* Pretty print an instruction list.
 *
 * @instr_list  The instruction list in question
 * @ninstr      Number of instructions
 * @txt         Text prepended the instruction list,
 *              ignored when NULL
 */
DLLEXPORT void bh_pprint_instr_list(const bh_instruction instr_list[],
                                    bh_intp ninstr, const char* txt);

DLLEXPORT void bh_pprint_instr_krnl(const bh_ir_kernel* krnl, const char* txt);

/* Pretty print an array view.
 *
 * @view  The array view in question
 */
DLLEXPORT void bh_pprint_array(const bh_view *view);

/* Pretty print an array base.
 *
 * @base  The array base in question
 */
DLLEXPORT void bh_pprint_base(const bh_base *base);

/* Pretty print an coordinate.
 *
 * @coord  The coordinate in question
 * @ndims  Number of dimensions
 */
DLLEXPORT void bh_pprint_coord(const bh_index coord[], bh_index ndims);

/* Pretty print an instruction trace of the BhIR.
 *
 * @bhir The BhIR in question
 *
 */
DLLEXPORT void bh_pprint_trace_file(const bh_ir *bhir, char trace_fn[]);

#ifdef __cplusplus
}
#endif
#endif
