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
  NARGS         = 3,
  NARRAY_ARGS   = 3,
  SYMBOL_TEXT   = ZIP-ADD-1D_2_1_0_~0Sd~1Cd~2Cd
}
*/
void KRN_52738296419217141553436208136659297258(operand_t** args, iterspace_t* iterspace)
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
    ///////////////////////////////////////////////////
    // Argument 2 - [ARRAY]
    //
    double* a2_first = ARGS_DPTR(2);
    const int64_t  a2_start  = args[2]->start;
    const int64_t  a2_nelem  = args[2]->nelem;
    const int64_t  a2_ndim   = args[2]->ndim;    
    int64_t*       a2_shape  = args[2]->shape;
    int64_t*       a2_stride = args[2]->stride;
    a2_first += a2_start;
    assert(a2_first != NULL);
    //
    // Operation(s)
    //
//
// Elementwise operation on one-dimensional arrays using strided indexing
{
    const int64_t nelements = iterspace->nelem;
    const int mthreads      = omp_get_max_threads();
    const int64_t nworkers  = nelements > mthreads ? mthreads : 1;
    const int64_t work_split= nelements / nworkers;
    const int64_t work_spill= nelements % nworkers;
    #pragma omp parallel num_threads(nworkers)
    {
        const int tid      = omp_get_thread_num();        // Thread info
        const int nthreads = omp_get_num_threads();
        int64_t work=0, work_offset=0, work_end=0;  // Work distribution
        if (tid < work_spill) {
            work = work_split + 1;
            work_offset = tid * work;
        } else {
            work = work_split;
            work_offset = tid * work + work_spill;
        }
        work_end = work_offset+work;
        if (work) {
        
        
        
        double* a0_current = a0_first + (work_offset *a0_stride[0]);
        
        
        
        double* a1_current = a1_first + (work_offset *a1_stride[0]);
        
        
        
        double* a2_current = a2_first + (work_offset *a2_stride[0]);
        for (int64_t i = work_offset; i < work_end; ++i) {
            *a2_current = *a1_current + *a0_current;
            
            a0_current += a0_stride[0];
            
            a1_current += a1_stride[0];
            
            a2_current += a2_stride[0];
            
        }
        }
    }
    // TODO: Handle write-out of non-temp and non-const scalars.
}
}
