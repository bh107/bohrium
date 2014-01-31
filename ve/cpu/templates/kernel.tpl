#ifndef BH_CPU_LIBS
#define BH_CPU_LIBS
#include <assert.h>
#include <stdarg.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <complex.h>
#include <math.h>
#include <Random123/threefry.h>
#if defined(_OPENMP)
#include <omp.h>
#else
inline int omp_get_max_threads() { return 1; }
inline int omp_get_thread_num()  { return 0; }
inline int omp_get_num_threads() { return 1; }
#endif
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define DEG_CIR 360.0
#define DEG_RAD (M_PI / (DEG_CIR / 2.0))
#define RAD_DEG ((DEG_CIR / 2.0) / M_PI)

#ifndef CPU_MISC
#define CPU_MAXDIM 16
#endif

typedef struct bh_kernel_args {
    int nargs;

    void*    data[30];
    int64_t  nelem[30];
    int64_t  ndim[30];
    int64_t  start[30];
    int64_t* shape[30];
    int64_t* stride[30];
} bh_kernel_args_t;

// hopefully this thing will be short-lived...
typedef struct { uint64_t start, key; } bh_r123;

void {{SYMBOL}}(bh_kernel_args_t* args)
{
    //
    // Argument unpacking
    //
    {{#ARGUMENT}}
    {{TYPE}} *a{{NR}}_first = args->data[{{NR}}];
    {{#ARRAY}}
    int64_t  a{{NR}}_nelem  = args->nelem[{{NR}}];
    int64_t  a{{NR}}_ndim   = args->ndim[{{NR}}];
    int64_t  a{{NR}}_start  = args->start[{{NR}}];
    int64_t *a{{NR}}_shape  = args->shape[{{NR}}];
    int64_t *a{{NR}}_stride = args->stride[{{NR}}];
    a{{NR}}_first += a{{NR}}_start;
    {{/ARRAY}}
    assert(a{{NR}}_first != NULL);
    {{/ARGUMENT}}
    
    //
    // Operation(s)
    //
    {{>OPERATIONS}}
}
