{{LICENSE}}
#ifndef KP_KERNEL_LIBS
#define KP_KERNEL_LIBS
#include <assert.h>
#include <stdarg.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <complex.h>
#include <limits.h>
#include <stdint.h>
#include <float.h>
#include <math.h>
#include <Random123/philox.h>
#if defined(_OPENMP)
#include <omp.h>
#else
static inline int omp_get_max_threads() { return 1; }
static inline int omp_get_thread_num()  { return 0; }
static inline int omp_get_num_threads() { return 1; }
#endif
#endif
#include "kp.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define DEG_CIR 360.0
#define DEG_RAD (M_PI / (DEG_CIR / 2.0))
#define RAD_DEG ((DEG_CIR / 2.0) / M_PI)

#ifndef KP_KERNEL_MISC
#define KP_KERNEL_MISC 1
#define CPU_MAXDIM 16
#endif

#ifndef KP_CODEGEN_MISC
#define KP_CODEGEN_MISC 1
typedef union philox2x32_as_1x64 {
    philox2x32_ctr_t orig;
    uint64_t combined;
} philox2x32_as_1x64_t;
#endif

/*
KERNEL-DESCRIPTION {
  MODE          = {{MODE}},
  LAYOUT        = {{LAYOUT}},
  NINSTR        = {{NINSTR}},
  NARRAY_INSTR  = {{NARRAY_INSTR}},
  NARGS         = {{NARGS}},
  NARRAY_ARGS   = {{NARRAY_ARGS}},
  OMASK         = {{OMASK}},
  SYMBOL_TEXT   = {{SYMBOL_TEXT}}
}
*/
void KRN_{{SYMBOL}}(kp_buffer** buffers, kp_operand** args, const kp_iterspace* const iterspace, const int offload_devid)
{
    //
    // Buffer unpacking
    {{BUFFERS}}

    //
    // Argument unpacking
    {{ARGUMENTS}}

    //
    // Iterspace unpacking
    {{ITERSPACE}}

    //
    // Offloading
    const int offload = offload_devid >= 0;

    //
    // Operation(s)
    //
    {{WALKER}}
}
