#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <inttypes.h>
#include "/home/safl/Desktop/bohrium/ve/cpu/timevault.cpp"

typedef struct operand {
    float* data;
    float* current;
    uint64_t nelements;
} operand_t;

using namespace bohrium::core;

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

/**
 *  This stuff ought to be similar to BH_IDENTITY.
 */
void with_fusion_id_2(operand_t* array)
{
    array->current = array->data;
    for(uint64_t idx=0; idx<array->nelements; ++idx) {
        *(array->current) = (float)1;
        *(array->current) = (float)2;
        (array->current)++;
    }
}

void with_fusion_id_4(operand_t* array)
{
    array->current = array->data;
    for(uint64_t idx=0; idx<array->nelements; ++idx) {
        *(array->current) = (float)1;
        *(array->current) = (float)2;
        *(array->current) = (float)1;
        *(array->current) = (float)2;
        (array->current)++;
    }
}

void without_fusion_id_2(operand_t* array)
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

void without_fusion_id_4(operand_t* array)
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

typedef void (*func)(operand_t*);

int main(int argc, char *argv[])
{
    const char usage[] = "usage: ./fusion.unary [y|n] 1000 10";
    if (2>argc) {
        fprintf(stderr, "%s\n", usage);
        return 1;
    }

    char* fuseit    = argv[1];
    const int ops         = atoi(argv[2]);
    const int size        = atoi(argv[3]);
    const int iterations  = atoi(argv[4]);

    func function;

    printf("%s %d %d\n", fuseit, size, iterations);

    operand_t array = setup(size);

    if (fuseit[0] == 'y') {
        switch(ops) {
            case 2:
                function = with_fusion_id_2;
                break;
            case 4:
                function = with_fusion_id_4;
                break;
            default:
                return 0;
        }
    } else if (fuseit[0] == 'n') {
        switch(ops) {
            case 2:
                function = without_fusion_id_2;
                break;
            case 4:
                function = without_fusion_id_4;
                break;
            default:
                return 0;
        }
    } else {
        printf("crappy...\n");
        return 0;
    }

    TIMER_START
    for(uint64_t i=0; i<iterations; i++) {
        function(&array);
    }
    TIMER_STOP("LOOP")

    return 0;
}

