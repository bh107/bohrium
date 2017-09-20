// Help function for the Fortran compiled kernels


#include<stdio.h>
#include<stdint.h>

void* convert_pointer(void* data_list[], int* idx) {
    return data_list[*idx];
}
