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
  MODE          = {{MODE}},
  LAYOUT        = {{LAYOUT}},
  NINSTR        = {{NINSTR}},
  NARRAY_INSTR  = {{NARRAY_INSTR}},
  NARGS         = {{NARGS}},
  NARRAY_ARGS   = {{NARRAY_ARGS}},
  SYMBOL_TEXT   = {{SYMBOL_TEXT}}
}
*/
void KRN_{{SYMBOL}}(operand_t** args, iterspace_t* iterspace)
{
    //
    // Argument unpacking
    //
    {{#ARGUMENT}}
    ///////////////////////////////////////////////////
    // Argument {{NR}} - [{{#SCALAR}}SCALAR{{/SCALAR}}{{#SCALAR_CONST}}SCALAR_CONST{{/SCALAR_CONST}}{{#SCALAR_TEMP}}SCALAR_TEMP{{/SCALAR_TEMP}}{{#ARRAY}}ARRAY{{/ARRAY}}]
    //
    {{#SCALAR}}
    {{TYPE}}* a{{NR}}_first = ARGS_DPTR({{NR}});
    assert(a{{NR}}_first != NULL);
    {{/SCALAR}}

    {{#SCALAR_CONST}}
    {{TYPE}}* a{{NR}}_first = ARGS_DPTR({{NR}});
    assert(a{{NR}}_first != NULL);
    {{/SCALAR_CONST}}

    {{#ARRAY}}
    {{TYPE}}* a{{NR}}_first = ARGS_DPTR({{NR}});
    const int64_t  a{{NR}}_start  = args[{{NR}}]->start;
    const int64_t  a{{NR}}_nelem  = args[{{NR}}]->nelem;
    const int64_t  a{{NR}}_ndim   = args[{{NR}}]->ndim;    
    int64_t*       a{{NR}}_shape  = args[{{NR}}]->shape;
    int64_t*       a{{NR}}_stride = args[{{NR}}]->stride;

    a{{NR}}_first += a{{NR}}_start;
    assert(a{{NR}}_first != NULL);
    {{/ARRAY}}

    {{/ARGUMENT}}
    
    //
    // Operation(s)
    //
    {{>OPERATIONS}}
}
