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
  MODE          = FUSED,
  LAYOUT        = STRIDED,
  NINSTR        = 72,
  NARRAY_INSTR  = 30,
  NARGS         = 22,
  NARRAY_ARGS   = 21,
  SYMBOL_TEXT   = ZIP-ADD-1D_1_0_0_ZIP-ADD-1D_2_1_0_ZIP-ADD-1D_0_0_2_ZIP-ADD-1D_3_0_0_ZIP-ADD-1D_4_3_0_ZIP-ADD-1D_0_0_4_ZIP-ADD-1D_5_0_0_ZIP-ADD-1D_6_5_0_ZIP-ADD-1D_0_0_6_ZIP-ADD-1D_7_0_0_ZIP-ADD-1D_8_7_0_ZIP-ADD-1D_0_0_8_ZIP-ADD-1D_9_0_0_ZIP-ADD-1D_10_9_0_ZIP-ADD-1D_0_0_10_ZIP-ADD-1D_11_0_0_ZIP-ADD-1D_12_11_0_ZIP-ADD-1D_0_0_12_ZIP-ADD-1D_13_0_0_ZIP-ADD-1D_14_13_0_ZIP-ADD-1D_0_0_14_ZIP-ADD-1D_15_0_0_ZIP-ADD-1D_16_15_0_ZIP-ADD-1D_0_0_16_ZIP-ADD-1D_17_0_0_ZIP-ADD-1D_18_17_0_ZIP-ADD-1D_0_0_18_ZIP-ADD-1D_19_0_0_ZIP-ADD-1D_20_19_0_ZIP-ADD-1D_0_0_20_~0Sd~1Td~2Cd~3Td~4Td~5Td~6Td~7Td~8Td~9Td~10Td~11Td~12Td~13Td~14Td~15Td~16Td~17Td~18Td~19Td~20Td~21Cd
}
*/
void KRN_423291310732022168912564607698417194944(operand_t** args, iterspace_t* iterspace)
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
    // Argument 1 - [SCALAR_TEMP]
    //
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
    ///////////////////////////////////////////////////
    // Argument 3 - [SCALAR_TEMP]
    //
    ///////////////////////////////////////////////////
    // Argument 4 - [SCALAR_TEMP]
    //
    ///////////////////////////////////////////////////
    // Argument 5 - [SCALAR_TEMP]
    //
    ///////////////////////////////////////////////////
    // Argument 6 - [SCALAR_TEMP]
    //
    ///////////////////////////////////////////////////
    // Argument 7 - [SCALAR_TEMP]
    //
    ///////////////////////////////////////////////////
    // Argument 8 - [SCALAR_TEMP]
    //
    ///////////////////////////////////////////////////
    // Argument 9 - [SCALAR_TEMP]
    //
    ///////////////////////////////////////////////////
    // Argument 10 - [SCALAR_TEMP]
    //
    ///////////////////////////////////////////////////
    // Argument 11 - [SCALAR_TEMP]
    //
    ///////////////////////////////////////////////////
    // Argument 12 - [SCALAR_TEMP]
    //
    ///////////////////////////////////////////////////
    // Argument 13 - [SCALAR_TEMP]
    //
    ///////////////////////////////////////////////////
    // Argument 14 - [SCALAR_TEMP]
    //
    ///////////////////////////////////////////////////
    // Argument 15 - [SCALAR_TEMP]
    //
    ///////////////////////////////////////////////////
    // Argument 16 - [SCALAR_TEMP]
    //
    ///////////////////////////////////////////////////
    // Argument 17 - [SCALAR_TEMP]
    //
    ///////////////////////////////////////////////////
    // Argument 18 - [SCALAR_TEMP]
    //
    ///////////////////////////////////////////////////
    // Argument 19 - [SCALAR_TEMP]
    //
    ///////////////////////////////////////////////////
    // Argument 20 - [SCALAR_TEMP]
    //
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
        
        
        double a1_current;
        
        
        
        
        double* a2_current = a2_first + (work_offset *a2_stride[0]);
        
        
        double a3_current;
        
        
        
        double a4_current;
        
        
        
        double a5_current;
        
        
        
        double a6_current;
        
        
        
        double a7_current;
        
        
        
        double a8_current;
        
        
        
        double a9_current;
        
        
        
        double a10_current;
        
        
        
        double a11_current;
        
        
        
        double a12_current;
        
        
        
        double a13_current;
        
        
        
        double a14_current;
        
        
        
        double a15_current;
        
        
        
        double a16_current;
        
        
        
        double a17_current;
        
        
        
        double a18_current;
        
        
        
        double a19_current;
        
        
        
        double a20_current;
        
        for (int64_t i = work_offset; i < work_end; ++i) {
             a1_current = *a0_current + *a0_current;
            *a2_current =  a1_current + *a0_current;
            *a0_current = *a0_current + *a2_current;
             a3_current = *a0_current + *a0_current;
             a4_current =  a3_current + *a0_current;
            *a0_current = *a0_current +  a4_current;
             a5_current = *a0_current + *a0_current;
             a6_current =  a5_current + *a0_current;
            *a0_current = *a0_current +  a6_current;
             a7_current = *a0_current + *a0_current;
             a8_current =  a7_current + *a0_current;
            *a0_current = *a0_current +  a8_current;
             a9_current = *a0_current + *a0_current;
             a10_current =  a9_current + *a0_current;
            *a0_current = *a0_current +  a10_current;
             a11_current = *a0_current + *a0_current;
             a12_current =  a11_current + *a0_current;
            *a0_current = *a0_current +  a12_current;
             a13_current = *a0_current + *a0_current;
             a14_current =  a13_current + *a0_current;
            *a0_current = *a0_current +  a14_current;
             a15_current = *a0_current + *a0_current;
             a16_current =  a15_current + *a0_current;
            *a0_current = *a0_current +  a16_current;
             a17_current = *a0_current + *a0_current;
             a18_current =  a17_current + *a0_current;
            *a0_current = *a0_current +  a18_current;
             a19_current = *a0_current + *a0_current;
             a20_current =  a19_current + *a0_current;
            *a0_current = *a0_current +  a20_current;
            
            a0_current += a0_stride[0];
            
            a2_current += a2_stride[0];
            
        }
        }
    }
    // TODO: Handle write-out of non-temp and non-const scalars.
}
}
