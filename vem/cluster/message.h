/*
 * Copyright 2011 Mads R. B. Kristensen <madsbk@gmail.com>
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
 * along with cphVB. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef __CPHVB_VEM_CLUSTER_MESSAGE_H
#define __CPHVB_VEM_CLUSTER_MESSAGE_H

#include <cphvb_type.h>
#include <cphvb_instruction.h>

//Message types
enum msg_types {CLUSTER_INST, CLUSTER_SHUTDOWN, CLUSTER_ARRAY};

//Communication message.
typedef struct
{
    //Message type. (e.g. CLUSTER_INST, CLUSTER_SHUTDOWN, etc.)
    cphvb_intp type;
}cluster_msg;

 //Communication message for instructions.
typedef struct
{
    //Message type, which is CLUSTER_INST in this case.
    cphvb_intp type;
    //Number of instruction in batch.
    cphvb_intp count;
    //The instruction list.
    cphvb_instruction inst[CPHVB_MAX_NO_INST];
}cluster_msg_inst;

//Communication message.
typedef struct
{
    //Message type, which is CLUSTER_ARRAY in this case.
    cphvb_intp type;
    //The new array.
    cphvb_array array;
}cluster_msg_array;



#define CLUSTER_MSG_SIZE (sizeof(cluster_msg_inst))
/*
//Make sure that all message types is smaller than CLUSTER_MSG_SIZE.
#if sizeof(cluster_msg_array) > CLUSTER_MSG_SIZE
#   error cluster_msg_array is greater than CLUSTER_MSG_SIZE
#endif
*/

#endif
