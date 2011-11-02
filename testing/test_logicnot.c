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
    
    printf("TESTING LOGICNOT\n");

    cphvb_error error;
    //We are using the vem_node VEM
    vem_init = &cphvb_vem_node_init;
    vem_execute = &cphvb_vem_node_execute;
    vem_shutdown = &cphvb_vem_node_shutdown;
    vem_create_array = &cphvb_vem_node_create_array;
    vem_instruction_check = &cphvb_vem_node_instruction_check;

    //initialize VEM
    error = vem_init();
    if(error != CPHVB_SUCCESS)
    {
        printf("Error in vem_init()\n");
        exit(-1);
    }
    

    //Create arrays (metadata only)
    cphvb_array *Ia, *Ib, *R;
    error =  vem_create_array(NULL,                       // base 
                              CPHVB_INT32,              // type
                              1,                          // ndim
                              0,                          // start
                              (cphvb_index[]){DATA_SIZE}, //shape
                              (cphvb_index[]){1},         //stride
                              0,                          //has_init_value
                              (cphvb_constant)0L,         //init_value
                              &Ia);
    error |=  vem_create_array(NULL, CPHVB_INT32, 1, 0,
                              (cphvb_index[]){DATA_SIZE},
                              (cphvb_index[]){1},
                              0, (cphvb_constant)0L, &Ib);
    error |=  vem_create_array(NULL, CPHVB_INT32, 1, 0,
                              (cphvb_index[]){DATA_SIZE},
                              (cphvb_index[]){1},
                              0, (cphvb_constant)0L, &R);
    if(error != CPHVB_SUCCESS)
    {
        printf("Error creating arrays\n");
        exit(-1);
    }
    
    //  Allocate mem for the two input arrays. 
    //  Since we want intialize them with data
    error = cphvb_malloc_array_data(Ia);
    error |= cphvb_malloc_array_data(Ib);
    if(error != CPHVB_SUCCESS)
    {
        printf("Error allocation memory for arrays\n");
        exit(-1);
    }

    for(int i=0; i<DATA_SIZE; ++i) {    // Fill arrays with known values
        ((int*)Ia->data)[i] = 1;
        ((int*)Ib->data)[i] = 1;
    }

    cphvb_instruction inst;             //Create the instruction (ADD)
    inst.opcode = CPHVB_LOGICAL_NOT;
    inst.operand[0] = R;
    inst.operand[1] = Ia;
    inst.operand[2] = Ib;

    if (vem_instruction_check(&inst))   //Check that the instruction is supported
    {                                   //Tell the VEM to perform the instruction.
        error = vem_execute(1,          // One instruction in the "batch
                            &inst);     // The "batch"
        if(error != CPHVB_SUCCESS)
        {
            printf("Error executing LOGICAL_ADD instruction\n");
            exit(-1);
        } 
    } 
    else                                // We should do it our selves 
    {                                   // This is left as an exercise ;-)
        printf("Throwing up: Operation LOGICAL_ADD not supported\n");
        exit(-1);
    }

    //Generate a sync instruction so we can se the result
    // CPHVB_SYNC    == read access
    // CPHVB_RELEASE == write access
    // CPHVB_RELEASE == CPHVB_SYNC + CPHVB_DISCARD
    cphvb_instruction inst_sync;
    //Tell the VEM to release the data to us.
    inst_sync.opcode = CPHVB_SYNC;
    inst_sync.operand[0] = R;
    error = vem_execute(1,&inst_sync);
    if(error != CPHVB_SUCCESS)
    {
        printf("Error executing SYNC instruction\n");
        exit(-1);
    }

    // Check that the result is as we expect
    int success = 1;
    for (int i = 0; i < DATA_SIZE; ++i)
    {
        //printf("[%f]", ((float*)R->data)[i]);
        if ((((int*)R->data)[i] + 1) != 1 )
        {
            success = 0;
            //break;
        }
    }
    if (success) 
        printf("LOGICAL_NOT: SUCCESS!\n");
    else
        printf("LOGICAL_NOT: FAIL! Calculation error!\n");

    //Tell the VEM we are done with the arrays
    void* datap[3];
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
    if(error != CPHVB_SUCCESS)
    {
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
