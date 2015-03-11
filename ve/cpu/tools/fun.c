#include <inttypes.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/mman.h>

#define CPU_MAXDIM 15

int main(void)
{
    int64_t ndim = 4;
    const int64_t last_dim = ndim-1;

    int64_t shape[CPU_MAXDIM];
    shape[3] = 150;
    shape[2] = 150;
    shape[1] = 150;
    shape[0] = 20;

    __asm__("# stride begin" : : );
    int64_t stride[CPU_MAXDIM];
    stride[last_dim] = 1;
    for(int64_t dim=last_dim-1; dim>=0; --dim) {
        stride[dim] = shape[dim+1];
    }
    __asm__("# stride end" : : );

    __asm__("# element begin" : : );
    int64_t nelements = 1;
    for(int64_t dim=0; dim<ndim; ++dim) {
        nelements *= shape[dim];
    }
    __asm__("# element end" : : );

    __asm__("# weight begin" : : );
    int64_t weight[CPU_MAXDIM];
    int64_t acc = 1;
    for(int64_t idx=last_dim; idx>=0; --idx) {
        weight[idx] = acc;
        acc *= shape[idx];
    }
    __asm__("# weight end" : : );
    
    double* array_a = (double*)malloc(sizeof(double)*nelements);    
    //double* array_a = mmap(0, size, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);

    int64_t chunksize = 100;
    for(int64_t eidx=0; eidx<nelements; eidx+=chunksize) {

        double* op = array_a;
        __asm__("# coord loop begin" : : );
        int64_t offset = 0;
        //#pragma omp simd reduction(+:offset)
        for(int64_t dim=0; dim<last_dim; ++dim) {
            //const int64_t coord = ;
            //const int64_t coord = (eidx / weight[dim]) - shape[dim] * (eidx/shape[dim]);
            //offset += coord * stride[dim];
            op += ((eidx / weight[dim]) % shape[dim]) * stride[dim];
        }
        __asm__("# coord loop end" : : );

        __asm__("# vec loop begin" : : );
        for(int64_t iidx=0; iidx<chunksize; ++iidx) {
            *op = 1.0;
            *op = *op * 2;
            *op = *op / 0.5;
            *op = *op / 0.5;
            *op = *op / 0.5;
            *op = *op / 0.5;
            ++op;
        }
        __asm__("# vec loop end" : : );
    }
    printf("blob=%f\n", *array_a);

}
