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

/* Pretty print an array.
 *
 * @view  The array view in question
 */
DLLEXPORT void bh_pprint_array(const bh_view *view);


/* Pretty print an instruction.
 *
 * @instr  The instruction in question
 */
DLLEXPORT void bh_pprint_instr(const bh_instruction *instr);

/* Pretty print an instruction list.
 *
 * @instr_list  The instruction list in question
 * @ninstr      Number of instructions
 * @txt         Text prepended the instruction list,
 *              ignored when NULL
 */
DLLEXPORT void bh_pprint_instr_list(const bh_instruction instr_list[],
                                    bh_intp ninstr, const char* txt);

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

/* Pretty print an BhIR DAG.
 *
 * @bhir The BhIR in question
 * @dag  The DAG in question
 *
 */
DLLEXPORT void bh_pprint_dag(const bh_ir *bhir, const bh_dag *dag);

/* Pretty print an BhIR.
 *
 * @bhir The BhIR in question
 *
 */
DLLEXPORT void bh_pprint_bhir(const bh_ir *bhir);

#ifdef __cplusplus
}
#endif

#endif
