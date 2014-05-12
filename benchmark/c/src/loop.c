#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <inttypes.h>

typedef struct operand {
    float* data;
    float* current;
    uint64_t nelements;
} operand_t;

operand_t setup(uint64_t nelements)
{
    operand_t array;

    array.data      = (float*)malloc(nelements*sizeof(float));    // Malloc
    array.current   = array.data;
    array.nelements = nelements;

    for(uint64_t i=0; i<nelements; i++) {
        *array.current = (float)1;
        array.current++;
    }

    return array;
}

void with_fusion_add(operand_t* array)
{
    array->current = array->data;
    for(uint64_t idx=0; idx<array->nelements; ++idx) {
        *(array->current) = *(array->current) + (float)1;
        *(array->current) = *(array->current) + (float)2;
        (array->current)++;
    }
}

void with_fusion_id(operand_t* array)
{
    array->current = array->data;
    for(uint64_t idx=0; idx<array->nelements; ++idx) {
        *(array->current) = (float)1;
        *(array->current) = (float)2;
        (array->current)++;
    }
}

void without_fusion_id(operand_t* array)
{
    array->current = array->data;
    for(uint64_t idx=0; idx<array->nelements; ++idx) {
        *(array->current) = (float)1;
        ++(array->current);
    }

    array->current = array->data;
    for(uint64_t idx=0; idx<array->nelements; ++idx) {
        *(array->current) = (float)2;
        ++(array->current);
    }
}

int main(int argc, char *argv[])
{
    operand_t array = setup(200000000);
    for(uint64_t i=0; i<10; i++) {
        //with_fusion_add(&array);
        //without_fusion_add(&array);

        //with_fusion_id(&array);
        without_fusion_id(&array);
    }
    return 0;
}

