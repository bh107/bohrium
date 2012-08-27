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
#ifndef __CPHVB_BRIDGE_CPP_STATE
#define __CPHVB_BRIDGE_CPP_STATE
#include <iostream>
#include "cphvb.h"

namespace cphvb {

#define CPHVB_CPP_QUEUE_MAX 1024
static cphvb_instruction queue[CPHVB_CPP_QUEUE_MAX]; // Instruction queue
static cphvb_intp queue_size = 0;

cphvb_init      vem_init;
cphvb_execute   vem_execute;
cphvb_shutdown  vem_shutdown;

cphvb_reg_func      vem_reg_func;
cphvb_component     **components,
                    *self_component,
                    *vem_component;
cphvb_intp          children_count;

void init()
{
    cphvb_error err;
    self_component = cphvb_component_setup();
    cphvb_component_children( self_component, &children_count, &components );

    if(children_count != 1 || components[0]->type != CPHVB_VEM) {

        fprintf(stderr, "Error in the configuration: the bridge must "
                        "have exactly one child of type VEM\n");
        exit(-1);
    }
    vem_component   = components[0];

    vem_init        = vem_component->init;
    vem_execute     = vem_component->execute;
    vem_shutdown    = vem_component->shutdown;

    vem_reg_func        = vem_component->reg_func;
    free(components);

    err = vem_init(vem_component);
    if(err) {
        fprintf(stderr, "Error in vem_init()\n");
        exit(-1);
    }

}

void shutdown()
{
    vem_shutdown();
    cphvb_component_free(self_component);
    cphvb_component_free(vem_component);
}

template <typename T>
inline
void enqueue_aaa( cphvb_opcode opcode, Vector<T> & op0, Vector<T> & op1, Vector<T> & op2)
{
    cphvb_instruction* instr;

    if (queue_size >= CPHVB_CPP_QUEUE_MAX) {
        vem_execute( queue_size, queue );
        queue_size = 0;
    }

    instr = &queue[queue_size++];
    instr->status = CPHVB_INST_PENDING;
    instr->opcode = opcode;
    instr->operand[0] = op0.array;
    instr->operand[1] = op1.array;
    instr->operand[2] = op2.array;

}

template <typename T>
inline
void enqueue_aac( cphvb_opcode opcode, Vector<T> & op0, Vector<T> & op1, T const& op2)
{
    cphvb_instruction* instr;

    if (queue_size >= CPHVB_CPP_QUEUE_MAX) {
        vem_execute( queue_size, queue );
        queue_size = 0;
    }

    instr = &queue[queue_size++];
    instr->status = CPHVB_INST_PENDING;
    instr->opcode = opcode;
    instr->operand[0] = op0.array;
    instr->operand[1] = op1.array;
    instr->operand[2] = NULL;
    assign_const_type( &instr->constant, op2 );

}

template <typename T>
inline
void enqueue_aca( cphvb_opcode opcode, Vector<T> & op0, T const& op1, Vector<T> & op2)
{
    cphvb_instruction* instr;

    if (queue_size >= CPHVB_CPP_QUEUE_MAX) {
        vem_execute( queue_size, queue );
        queue_size = 0;
    }

    instr = &queue[queue_size++];
    instr->status = CPHVB_INST_PENDING;
    instr->opcode = opcode;
    instr->operand[0] = op0.array;
    instr->operand[1] = NULL;
    instr->operand[2] = op2.array;
    assign_const_type( &instr->constant, op1 );

}

template <typename T>
inline
void enqueue_aa( cphvb_opcode opcode, Vector<T> & op0, Vector<T> & op1)
{
    cphvb_instruction* instr;

    if (queue_size >= CPHVB_CPP_QUEUE_MAX) {
        vem_execute( queue_size, queue );
        queue_size = 0;
    }

    instr = &queue[queue_size++];
    instr->status = CPHVB_INST_PENDING;
    instr->opcode = opcode;
    instr->operand[0] = op0.array;
    instr->operand[1] = op1.array;
    instr->operand[2] = NULL;

}

template <typename T>
inline
void enqueue_ac( cphvb_opcode opcode, Vector<T> & op0, T const& op1)
{
    cphvb_instruction* instr;

    if (queue_size >= CPHVB_CPP_QUEUE_MAX) {
        vem_execute( queue_size, queue );
        queue_size = 0;
    }

    instr = &queue[queue_size++];
    instr->status = CPHVB_INST_PENDING;
    instr->opcode = opcode;
    instr->operand[0] = op0.array;
    instr->operand[1] = NULL;
    instr->operand[2] = NULL;
    assign_const_type( &instr->constant, op1 );

}

//
// Helper functions for printing and stuff like that...
//
cphvb_intp flush()
{
    char *msg = (char*) malloc(1024 * sizeof(char));

    cphvb_intp cur_size = queue_size;
    cphvb_error res = CPHVB_SUCCESS;
    if (queue_size > 0) {
        res = vem_execute( queue_size, queue );

        if (res != CPHVB_SUCCESS) {
            sprintf(msg, "Error in scheduled batch of instructions: %s\n", cphvb_error_text(res));
            printf("%s", msg);
            for(int i=0; i<queue_size; i++) {
                sprintf(msg, "%s\n", cphvb_error_text( queue[i].status ));
                printf("%d-%s", i, msg);
                if ((queue[i].status != CPHVB_SUCCESS) && (queue[i].status != CPHVB_INST_PENDING)) {
                    cphvb_pprint_instr( &queue[i] );
                }
            }
        }

        queue_size = 0;
    }
    return cur_size;
   
}

}

#endif
