#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>

#include <cphvb.h>
#include <cphvb_vem.h>
#include <cphvb_vem_node.h>

#define DATA_SIZE 1024*1024*3

//Function pointers to the VEM.
static cphvb_vem_init vem_init;
static cphvb_vem_execute vem_execute;
static cphvb_vem_shutdown vem_shutdown;
static cphvb_vem_create_array vem_create_array;
static cphvb_vem_instruction_check vem_instruction_check;

int main (int argc, char** argv)
{
    
    printf("TESTING ADD vs INLINED ADD.\n");

    cphvb_error error;
    vem_init                = &cphvb_vem_node_init;
    vem_execute             = &cphvb_vem_node_execute;
    vem_shutdown            = &cphvb_vem_node_shutdown;
    vem_create_array        = &cphvb_vem_node_create_array;
    vem_instruction_check   = &cphvb_vem_node_instruction_check;

    cphvb_instruction batch[100];

    //initialize VEM
    error = vem_init();
    if(error != CPHVB_SUCCESS) {
        printf("Error in vem_init()\n");
        exit(-1);
    }

    //Create arrays (metadata only)
    cphvb_array *Ia, *Ib, *R;
    error =  vem_create_array(
        NULL,                       // base 
        CPHVB_FLOAT32,              // type
        1,                          // ndim
        0,                          // start
        (cphvb_index[]){DATA_SIZE}, //shape
        (cphvb_index[]){1},         //stride
        0,                          //has_init_value
        (cphvb_constant)0L,         //init_value
        &Ia
    );
    error |=  vem_create_array(NULL, CPHVB_FLOAT32, 1, 0,
                              (cphvb_index[]){DATA_SIZE},
                              (cphvb_index[]){1},
                              0, (cphvb_constant)0L, &Ib);
    error |=  vem_create_array(NULL, CPHVB_FLOAT32, 1, 0,
                              (cphvb_index[]){DATA_SIZE},
                              (cphvb_index[]){1},
                              0, (cphvb_constant)0L, &R);
    if(error != CPHVB_SUCCESS) {
        printf("Error creating arrays\n");
        exit(-1);
    }
    
    //  Allocate mem for the two input arrays. 
    //  Since we want intialize them with data
    error = cphvb_malloc_array_data(Ia);
    error |= cphvb_malloc_array_data(Ib);
    if(error != CPHVB_SUCCESS) {
        printf("Error allocation memory for arrays\n");
        exit(-1);
    }

    for(int i=0; i<DATA_SIZE; ++i) {    // Fill arrays with known values
        ((float*)Ia->data)[i] = 2.0;
        ((float*)Ib->data)[i] = 2.0;
    }

    batch[0].opcode = CPHVB_ADD;
    batch[0].operand[0] = R;
    batch[0].operand[1] = Ia;
    batch[0].operand[2] = Ib;
  
    batch[1].opcode = CPHVB_SYNC;
    batch[1].operand[0] = R;

    for(int i=0; i<100; i++) {

        error = vem_execute(2, batch);
        if(error != CPHVB_SUCCESS) {
            printf("Error execution instruction batch.\n");
            exit(-1);
        }
        
    }


    // Check that the result is as we expect
    int success = 1;
    for (int i = 0; i < DATA_SIZE; ++i) {
        float sum = ((float*)Ia->data)[i] + ((float*)Ib->data)[i]; 
        if ((((float*)R->data)[i] - sum) != 0.0 ) {
            success = 0;
        }
    }
    if (success) 
        printf("BATCH: SUCCESS!\n");
    else
        printf("BATCH: FAIL! Calculation error!\n");

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
