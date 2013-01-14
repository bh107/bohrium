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

#ifndef __CPHVB_VEM_CLUSTER_TASK_H
#define __CPHVB_VEM_CLUSTER_TASK_H

/* Codes for data types */
enum /* cphvb_type */
{
    TASK_INST,            //Regular cphVB instruction
    TASK_SEND_RECV,       //Send or receive p2p 
    TASK_GATHER_SCATTER,  //Gather or scatter collective
};
typedef cphvb_intp task_type;


//Local instruction task
typedef struct
{
    //The type of this task
    task_type type;
    //The local instruction that makes up this task
    cphvb_instruction inst;
}task_inst;


//Gather or Scatter task
typedef struct
{
    //The type of this task
    task_type type;
    //If True we scatter else we gather
    bool direction;
    //The global array to gather or scatter
    cphvb_array *global_ary;
}task_gather_scatter;


//Gather or Scatter task
typedef struct
{
    //The type of this task
    task_type type;
    //If True we send the array else we receive it
    bool direction;
    //The local array to send or receive
    cphvb_array *local_ary;
    //The process to send to or receive from
    int rank;
}task_send_recv;


typedef union
{
    task_inst            inst;
    task_gather_scatter  gather_scatter;
    task_send_recv       send_recv;
}task;


#endif
