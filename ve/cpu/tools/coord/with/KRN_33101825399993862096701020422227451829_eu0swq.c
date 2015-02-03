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
#include <Random123/philox.h>
#include <tac.h>
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
#ifndef CPU_CODEGEN_MISC
#define CPU_CODEGEN_MISC 1
#define ARGS_DPTR(I) *(args[I]->data)
typedef union philox2x32_as_1x64 {
    philox2x32_ctr_t orig;
    uint64_t combined;
} philox2x32_as_1x64_t;
#endif
/*
KERNEL-DESCRIPTION {
  MODE          = SIJ,
  LAYOUT        = INSPECT_THE_GENERATED_CODE,
  NINSTR        = 1,
  NARRAY_INSTR  = 1,
  NARGS         = 2,
  NARRAY_ARGS   = 2,
  SYMBOL_TEXT   = ZIP-ADD-1D_1_0_0_~0Sd~1Cd
}
*/
void KRN_33101825399993862096701020422227451829(operand_t** args, iterspace_t* iterspace)
{
    //
    // Argument unpacking
    //
    ///////////////////////////////////////////////////
    // Argument 0 - [ARRAY]
    //
    double* a0_first = ARGS_DPTR(0);
    const int64_t  a0_start  = args[0]->start;
    const int64_t  a0_nelem  = args[0]->nelem;
    const int64_t  a0_ndim   = args[0]->ndim;    
    int64_t*       a0_shape  = args[0]->shape;
    int64_t*       a0_stride = args[0]->stride;
    a0_first += a0_start;
    assert(a0_first != NULL);
    ///////////////////////////////////////////////////
    // Argument 1 - [ARRAY]
    //
    double* a1_first = ARGS_DPTR(1);
    const int64_t  a1_start  = args[1]->start;
    const int64_t  a1_nelem  = args[1]->nelem;
    const int64_t  a1_ndim   = args[1]->ndim;    
    int64_t*       a1_shape  = args[1]->shape;
    int64_t*       a1_stride = args[1]->stride;
    a1_first += a1_start;
    assert(a1_first != NULL);
    //
    // Operation(s)
    //
//
// Elementwise operation on strided arrays of any dimension/rank.
{
    const int64_t nelements = iterspace->nelem;
    const int64_t last_dim  = iterspace->ndim-1;
    const int64_t shape_ld  = iterspace->shape[last_dim];
    
    
    
    int64_t  a0_stride_ld = a0_stride[last_dim];
    
    
    
    int64_t  a1_stride_ld = a1_stride[last_dim];
    int64_t weight[CPU_MAXDIM];
    int acc = 1;
    for(int idx=iterspace->ndim-1; idx >=0; --idx) {
        weight[idx] = acc;
        acc *= iterspace->shape[idx];
    }
    #pragma omp parallel for schedule(static)
    for(int64_t eidx=0; eidx<nelements; ++eidx) {
        
        double* a0_current = a0_first;
        
        double* a1_current = a1_first;
        
        for (int64_t dim=0; dim <= last_dim; ++dim) {
            const int64_t dim_position = (eidx / weight[dim]) % iterspace->shape[dim];
            
            a0_current += dim_position * a0_stride[dim];
            
            a1_current += dim_position * a1_stride[dim];
            
        }
        *a1_current = *a0_current + *a0_current;
    }
    // TODO: Handle write-out of non-temp and non-const scalars.
}
}
