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

#include <bh.h>
#include "task.h"

#ifndef __BH_VEM_CLUSTER_BATCH_H
#define __BH_VEM_CLUSTER_BATCH_H



/* Schedule an task.
 * @t  The task to schedule
 */
void batch_schedule(const task& t);


/* Schedule an instruction
 * @inst   The instruction to schedule
 */
void batch_schedule_inst(const bh_instruction& inst);


/* Schedule an instruction that only takes one instruction.
 *
 * @opcode   The opcode of the instruction
 * @operand  The local base array in the instruction
 */
void batch_schedule_inst(bh_opcode opcode, bh_base *operand);


/* Schedule an instruction.
 *
 * @opcode   The opcode of the instruction
 * @operands The local operands in the instruction
 * @ufunc    The user-defined function struct when opcode is BH_USERFUNC.
 */
void batch_schedule_inst(bh_opcode opcode, bh_view *operands,
                         bh_userfunc *ufunc);


/* Schedule an send/receive instruction.
 *
 * @direction   If True the array is send else it is received.
 * @rank        The process to send to or receive from
 * @local_view  The local view to communicate
 *              NB: it must be contiguous (row-major)
 */
void batch_schedule_comm(bool direction, int rank, const bh_view &local_view);


/* Flush all scheduled instructions
 *
 */
void batch_flush();


#endif
