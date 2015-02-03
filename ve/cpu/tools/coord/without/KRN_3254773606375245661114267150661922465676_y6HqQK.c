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
  SYMBOL_TEXT   = MAP-IDENTITY-1D_1_0_~0Kd~1Cd
}
*/
void KRN_3254773606375245661114267150661922465676(operand_t** args, iterspace_t* iterspace)
{
    //
    // Argument unpacking
    //
    ///////////////////////////////////////////////////
    // Argument 0 - [SCALAR_CONST]
    //
    double* a0_first = ARGS_DPTR(0);
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
// Elementwise operation on contigous arrays of any dimension/rank.
//  * Collapses the loops for every dimension into a single loop.
//  * Simplified array-walking (++)
//  * TODO: Vectorization, alias/restrict
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
        
        const double a0_current = *a0_first;
        
        
        
        
        
        double* a1_current = a1_first + work_offset;
        for (int64_t i = work_offset; i<work_end; ++i) {
            *a1_current =  a0_current;
            
            ++a1_current;
            
        }
        }
    }
    // TODO: Handle write-out of non-temp and non-const scalars.
}
}
