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

#include <cphvb.h>
#include "task.h"

#ifndef __CPHVB_VEM_CLUSTER_BATCH_H
#define __CPHVB_VEM_CLUSTER_BATCH_H



/* Schedule an task. 
 * NB: for now, we will flush in every task scheduling
 * @t  The task to schedule 
 */
void batch_schedule(const task& t);


/* Schedule an instruction
 * @inst   The instruction to schedule 
 */
void batch_schedule(const cphvb_instruction& inst);


/* Schedule an instruction that only takes one instruction.
 *
 * @opcode   The opcode of the instruction
 * @operand  The local operand in the instruction
 */
void batch_schedule(cphvb_opcode opcode, cphvb_array *operand);


/* Schedule an instruction.
 *
 * @opcode   The opcode of the instruction
 * @operands The local operands in the instruction
 * @ufunc    The user-defined function struct when opcode is CPHVB_USERFUNC.
 */
void batch_schedule(cphvb_opcode opcode, cphvb_array *operands[],
                    cphvb_userfunc *ufunc);


/* Schedule an send/receive instruction.
 *
 * @direction  If True the array is send else it is received.
 * @rank       The to send to or receive from
 * @local_ary  The local array to communicate
 */
void batch_schedule(bool direction, int rank, cphvb_array *local_ary);


/* Flush all scheduled instructions
 * 
 */
void batch_flush();


#endif
