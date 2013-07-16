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

#ifndef __BH_VEM_CLUSTER_TASK_H
#define __BH_VEM_CLUSTER_TASK_H

/* Codes for data types */
enum /* bh_type */
{
    TASK_INST,            //Regular Bohrium instruction
    TASK_SEND_RECV,       //Send or receive p2p
};
typedef bh_intp task_type;


//Local instruction task
typedef struct
{
    //The type of this task
    task_type type;
    //The local instruction that makes up this task
    bh_instruction inst;
}task_inst;


//Gather or Scatter task
typedef struct
{
    //The type of this task
    task_type type;
    //If True we send the array else we receive it
    bool direction;
    //The local view to send or receive
	//NB: This view should never be seen by the rest of Bohrium 
	//    and thus never discarded. 
    //    Furthermore, it must be contiguous (row-major)
    bh_view local_view;
    //The process to send to or receive from
    int rank;
}task_send_recv;


typedef union
{
    task_inst            inst;
    task_send_recv       send_recv;
}task;


#endif
