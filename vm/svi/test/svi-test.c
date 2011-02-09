#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cphvb.h>
#include <svi.h>

void test_callback(cphvb_int32 batch_id,
                   cphvb_int32 instruction_count,
                   cphvb_error error_code)
{
    printf("[Test] Got callback: {batch_id: %d, instruction_count %d, error_code: %d}\n",batch_id,instruction_count, error_code);
}


int main(int argc, char** argv)
{

    int _1d = (int)cphvb_size(CPHVB_ADD, 1, 0);
    int _2d = (int)cphvb_size(CPHVB_ADD, 2, 0);
    int _3d = (int)cphvb_size(CPHVB_ADD, 3, 0);
    int _4d = (int)cphvb_size(CPHVB_ADD, 4, 0);
    
    printf("Size of 1d Add: %d bytes\n",_1d);
    printf("Size of 2d Add: %d bytes\n",_2d);
    printf("Size of 3d Add: %d bytes\n",_3d);
    printf("Size of 4d Add: %d bytes\n",_4d);
    
    cphvb_instruction* instruction_queue = malloc(sizeof(cphvb_instruction)*10);
    char* seri = malloc(_2d*20);

    int instid = 0;
    int error = -1;
    char* next_seri = seri;
    
    // _101 = empty(10, dtype=int32)
    cphvb_instruction* inst = &instruction_queue[instid];
    next_seri = cphvb_init(inst,CPHVB_MALLOC,1,0,next_seri);
    cphvb_set_shape(inst,(long[]){10});
    cphvb_set_operand(inst,0,101,CPHVB_INT32,0,NULL);
//    error = svi_do(inst->serialized);
    char* buf = (char*)malloc(1024);    
    cphvb_snprint(inst,1024,buf);
    printf("%s",buf);

    // _101.fill(18)
    inst = &instruction_queue[++instid];
    next_seri = cphvb_init(inst,CPHVB_ADD,1,2,next_seri);
    cphvb_set_shape(inst,(long[]){10});
    cphvb_set_operand(inst,0,101,CPHVB_INT32,0,(long[]){1});
    cphvb_set_constant(inst,1,(cphvb_constant)0,CPHVB_INT32);
    cphvb_set_constant(inst,2,(cphvb_constant)18,CPHVB_INT32);
//    error = svi_do(inst->serialized);
    cphvb_snprint(inst,1024,buf);
    printf("%s",buf);

    // _102 = empty((8), dtype=int32)
    inst = &instruction_queue[++instid];
    next_seri = cphvb_init(inst,CPHVB_MALLOC,1,0,next_seri);
    cphvb_set_shape(inst,(long[]){8});
    cphvb_set_operand(inst,0,102,CPHVB_INT32,0,NULL);
//    error = svi_do(inst->serialized);
    cphvb_snprint(inst,1024,buf);
    printf("%s",buf);


    // _102.fill(24)
    inst = &instruction_queue[++instid];
    next_seri = cphvb_init(inst,CPHVB_ADD,1,2,next_seri);
    cphvb_set_shape(inst,(long[]){8});
    cphvb_set_operand(inst,0,102,CPHVB_INT32,0,(long[]){1});
    cphvb_set_constant(inst,1,(cphvb_constant)0,CPHVB_INT32);
    cphvb_set_constant(inst,2,(cphvb_constant)24,CPHVB_INT32);
//    error = svi_do(inst->serialized);
    cphvb_snprint(inst,1024,buf);
    printf("%s",buf);



    // _103 = empty((8,10), dtype=int32)
    inst = &instruction_queue[++instid];
    next_seri = cphvb_init(inst,CPHVB_MALLOC,2,0,next_seri);
    cphvb_set_shape(inst,(long[]){8,10});
    cphvb_set_operand(inst,0,103,CPHVB_INT32,0,(long[]){10,1});
//    error = svi_do(inst->serialized);
    cphvb_snprint(inst,1024,buf);
    printf("%s",buf);
//    printf("%d:%s",error,buf);


    // _103 = _101 + _102
    inst = &instruction_queue[++instid];
    next_seri = cphvb_init(inst,CPHVB_ADD,2,0,next_seri);
    cphvb_set_shape(inst,(long[]){8,10});
    cphvb_set_operand(inst,0,103,CPHVB_INT32,0,(long[]){10,1});
    cphvb_set_operand(inst,1,101,CPHVB_INT32,0,(long[]){0,1});
    cphvb_set_operand(inst,2,102,CPHVB_INT32,0,(long[]){1,0});
//    error = svi_do(inst->serialized);
    cphvb_snprint(inst,1024,buf);
    printf("%s",buf);

    

    cphvb_int32* a1; 
    inst = &instruction_queue[++instid];
    next_seri = cphvb_init(inst,CPHVB_READ,2,0,next_seri);
    cphvb_set_shape(inst,(long[]){8,10});
    cphvb_set_operand(inst,0,103,CPHVB_INT32,0,(long[]){10,1});
    cphvb_set_constant(inst,1,(cphvb_constant)(cphvb_ptr)&a1,CPHVB_PTR);
//    error = svi_do(inst->serialized);
    cphvb_snprint(inst,1024,buf);
    printf("%s",buf);

    error = svi_init(&test_callback);
    printf("[Test] svi_init: %d\n",error);

    error = svi_execute(7654,instid+1,seri);
    printf("[Test] svi_execute: %d\n",error);


    int i,j;
    printf("[\n");
    for (i = 0; i<8; ++i)
    {
        printf("  [%d",*(a1+8*i));
        for (j = 1; j<10; ++j)
        {
            printf(", %d",*(a1+8*i+j));
        }
        printf("],\n");
    }
    printf("]\n");


    return 0;
}
