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

#ifndef __BH_IR_MAP_H
#define __BH_IR_MAP_H

#include "bh_error.h"

#ifdef __cplusplus
extern "C" {
#endif

//The function type for the map instruction
typedef bh_error (*bh_ir_map_instr_func)(bh_instruction *instr);

/* Applies the 'func' on each instruction in the 'dag' and it's
 * sub-dags topologically.
 *
 * @bhir        The BhIR handle
 * @dag         The dag to start the map from
 * @func        The func to call with all instructions
 * @return      Error code (BH_SUCCESS, BH_ERROR)
 */
DLLEXPORT bh_error bh_ir_map_instr(bh_ir *bhir, bh_dag *dag,
                         bh_ir_map_instr_func func);

#ifdef __cplusplus
}
#endif

#endif

