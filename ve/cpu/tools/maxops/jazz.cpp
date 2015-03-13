#include <inttypes.h>
#include <stdlib.h>
#include <stdio.h>

typedef void (*alloc_func)(uint32_t narrays, uint32_t nelements, double** arrays);
typedef void (*init_func)(uint32_t narrays, uint32_t nelements, double** arrays);
typedef void (*apply_func)(uint32_t narrays, uint32_t nelements, double** arrays);
typedef void (*free_func)(uint32_t narrays, double** arrays);
typedef void (*print_func)(uint32_t narrays, uint32_t nelements, double**);

void alloc_n(uint32_t narrays, uint32_t nelements, double** arrays)
{
    for(uint32_t idx=0; idx<narrays; ++idx) {
        arrays[idx] = (double*)malloc(nelements*sizeof(double));
    }
}

void init_3(uint32_t narrays, uint32_t nelements, double** arrays)
{
    for(uint32_t aidx=0; aidx<narrays; ++aidx) {
        double* operand = arrays[aidx];
        for(uint32_t eidx=0; eidx<nelements; ++eidx) {
            *operand = 1.0;
            ++operand;
        }
    }
}

void free_3(uint32_t narrays, double** arrays)
{
    for(uint32_t aidx=0; aidx<narrays; ++aidx) {
        double* operand = arrays[aidx];
        free(operand);
    }
}

void apply_3(uint32_t nelements, double* array_1, double* array_2, double* array_3)
{
    double* op1 = array_1;
    double* op2 = array_2;
    double* op3 = array_3;

    for(uint32_t eidx=0; eidx<nelements; ++eidx) {
        *op1 = *op2 + *op3;

        ++op1; ++op2; ++op3;
    }
}

void print_3(uint32_t nelements, double* array_1, double* array_2, double* array_3)
{
    printf("array_1 = %f, array_2 = %f, array_3 = %f\n", *array_1, *array_2, *array_3);
    printf("array_1 = %f, array_2 = %f, array_3 = %f\n",
        *(array_1+nelements-1), *(array_2+nelements-1), *(array_3+nelements-1)
    );
    printf("array_1 = %f, array_2 = %f, array_3 = %f\n",
        *(array_1+nelements/2), *(array_2+nelements/2), *(array_3+nelements/2)
    );
}

void experiment_3(void)
{
    alloc_func allocator    = alloc_3;
    init_func initializer   = init_3;
    apply_func applicator   = apply_3;
    print_func printer      = print_3;
    free_func deallocator   = free_3;
    
    double* array_1 = NULL;
    double* array_2 = NULL;
    double* array_3 = NULL;

    uint32_t nelements    = 50000000;
    uint32_t iterations   = 10;
    uint32_t narrays      = 10;

    double* arrays[narrays];

    allocator(nelements, &array_1, &array_2, &array_3);

    initializer(nelements, array_1, array_2, array_3);

    for(uint32_t it=0; it<iterations; ++it) {
        applicator(nelements, array_1, array_2, array_3);
    }

    printer(nelements, array_1, array_2, array_3);
    deallocator(array_1, array_2, array_3);
}

int main(void)
{
    experiment_3();
    return 0;
}
