/*
 * Copyright 2011 Troels Blum <troels@blum.dk>
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

/*
 * This is a simple example, demonstrating how to connect to the VEM, 
 * and execute a single operarion.
 */

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>

#include <cphvb.h>
#include <cphvb_vem.h>
#include <cphvb_vem_node.h>

#define DATA_SIZE 1024

//Function pointers to the VEM.
static cphvb_vem_init vem_init;
static cphvb_vem_execute vem_execute;
static cphvb_vem_shutdown vem_shutdown;
static cphvb_vem_create_array vem_create_array;
static cphvb_vem_instruction_check vem_instruction_check;

int main (int argc, char** argv)
{

    printf("TESTING RSHIFT\n");
    cphvb_error error;
                                        // We are using the vem_node VEM
    vem_init                = &cphvb_vem_node_init;
    vem_execute             = &cphvb_vem_node_execute;
    vem_shutdown            = &cphvb_vem_node_shutdown;
    vem_create_array        = &cphvb_vem_node_create_array;
    vem_instruction_check   = &cphvb_vem_node_instruction_check;

    
    error = vem_init();                 // Initialize VEM
    if(error != CPHVB_SUCCESS) {
        printf("Error in vem_init()\n");
        exit(-1);
    }
    

                                        // Create arrays (metadata only)
    cphvb_array *Ia, *Ib, *R;
    error =  vem_create_array(
        NULL,                       // base 
        CPHVB_INT32,              // type
        1,                          // ndim
        0,                          // start
        (cphvb_index[]){DATA_SIZE}, //shape
        (cphvb_index[]){1},         //stride
        0,                          //has_init_value
        (cphvb_constant)0L,         //init_value
        &Ia
    );
    error |=  vem_create_array(
        NULL, CPHVB_INT32, 1, 0,
        (cphvb_index[]){DATA_SIZE},
        (cphvb_index[]){1},
        0, (cphvb_constant)0L, &Ib
    );
    error |=  vem_create_array(
        NULL, CPHVB_INT32, 1, 0,
        (cphvb_index[]){DATA_SIZE},
        (cphvb_index[]){1},
        0, (cphvb_constant)0L, &R
    );

    if(error != CPHVB_SUCCESS) {
        printf("Error creating arrays\n");
        exit(-1);
    }
    
    error = cphvb_malloc_array_data(Ia);                // Allocate mem for the two input arrays. 
    error |= cphvb_malloc_array_data(Ib);               // Since we want to intialize them with data
    if(error != CPHVB_SUCCESS) {
        printf("Error allocation memory for arrays\n");
        exit(-1);
    }

    for(int i=0; i<DATA_SIZE; ++i) {
        ((int*)Ia->data)[i] = -1.0;
        ((int*)Ib->data)[i] = 1.0;
    }

    cphvb_instruction inst;                             // Create the instruction
    inst.opcode = CPHVB_RIGHT_SHIFT;
    inst.operand[0] = R;
    inst.operand[1] = Ia;
    inst.operand[2] = Ib;
                                                        // Check that the instruction is supported
    if (vem_instruction_check(&inst)) {                 //  Tell the VEM to perform the instruction.

        error = vem_execute(1, &inst);                  // One instruction in the "batch
        if(error != CPHVB_SUCCESS) {
            printf("Error executing instruction\n");
            exit(-1);
        } 
    } else {
        printf("Throwing up: Operation not supported\n");
        exit(-1);
    }

    cphvb_instruction inst_sync;                        // Generate a sync instruction so we can se the result
    inst_sync.opcode        = CPHVB_SYNC;               //  CPHVB_SYNC    == read access
    inst_sync.operand[0]    = R;                        //  CPHVB_RELEASE == write access
    error = vem_execute(1,&inst_sync);                  //  CPHVB_RELEASE == CPHVB_SYNC + CPHVB_DISCARD
                                                        // Tell the VEM to release the data to us.
    if(error != CPHVB_SUCCESS) {
        printf("Error executing SYNC instruction\n");
        exit(-1);
    }

    int success = 1;                                    // Check that the result is as we expect
    for (int i = 0; i < DATA_SIZE; ++i) {

        if (((((int*)R->data)[i]) + 1) != 0.00) {
            printf("Val = %d\n", (((int*)R->data)[i]));
            success = 0;
            break;
        }
    }
    if (success) { 
        printf("LSHIFT: SUCCESS!\n");
    } else {
        printf("LSHIFT: FAIL! Calculation error!\n");
    }

    void* datap[3];                                     // Tell the VEM we are done with the arrays
    datap[0] = Ia->data;
    datap[1] = Ib->data;
    datap[2] = R->data;
    cphvb_instruction inst_destroy[3];
    inst_destroy[0].opcode = CPHVB_DESTROY;
    inst_destroy[0].operand[0] = Ia;
    inst_destroy[1].opcode = CPHVB_DESTROY;
    inst_destroy[1].operand[0] = Ib;
    inst_destroy[2].opcode = CPHVB_DESTROY;
    inst_destroy[2].operand[0] = R;
    error = vem_execute(3,inst_destroy);
    if(error != CPHVB_SUCCESS) {
        printf("Error executing DESTROY instructions\n");
        exit(-1);
    }

    // Free the memory for good measure
    free(datap[0]);
    free(datap[1]);
    free(datap[2]);

    //And we are done
    return 0;
    
}
