#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main() {
    int iterations = 10;
    int nelements = 1000*1000*10;
    double* a = (double*)malloc(sizeof(double)*nelements);
    double* b = (double*)malloc(sizeof(double)*nelements);
    double* c = (double*)malloc(sizeof(double)*nelements);

    for(int i=0; i<nelements; i++) {
        a[i] = i;
        b[i] = i;
        c[i] = 1;
    }

    for(int iter=0; iter<iterations; iter++) {
        #pragma omp parallel for simd schedule(static)
        for(int i=0; i<nelements; i++) {
            a[i] = b[i]+iter + c[i]; 
            b[i] = a[i]+iter + c[i]; 
            
            c[i] = pow(a[i], ((long)b[i]) % 2) ;
            //c[i] = a[i] + b[i]*iter;
        }
    }
    /*
    double sum = 0;
    for(int i=0; i<nelements; i++) {
        sum += c[i];
    }*/
    printf("sum= %lf\n", c[1000]);

    return 0;
}
