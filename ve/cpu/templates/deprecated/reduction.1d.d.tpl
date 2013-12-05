{{#license}}
/*
This file is part of Bohrium and copyright (c) 2012 the Bohrium
team <http://www.bh107.org>.

Bohrium is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as 
published by the Free Software Foundation, either version 3 
of the License, or (at your option) any later version.

Bohrium is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the 
GNU Lesser General Public License along with Bohrium. 

If not, see <http://www.gnu.org/licenses/>.
*/
{{/license}}
{{#include}}
#include "assert.h"
#include "stdarg.h"
#include "string.h"
#include "stdlib.h"
#include "stdint.h"
#include "stdio.h"
#include "complex.h"
#include "math.h"

#include "omp.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define DEG_CIR 360.0
#define DEG_RAD (M_PI / (DEG_CIR / 2.0))
#define RAD_DEG ((DEG_CIR / 2.0) / M_PI)

#ifndef DYNAMITE_MISC
#define DYNAMITE_MAXDIM 16
#endif
{{/include}}

/*
int reduction(
    int tool,

    T       *a0_first,
    int64_t  a0_start,
    int64_t *a0_stride,
    int64_t *a1_shape,
    int64_t  a1_ndim,

    T       *a1_first,
    int64_t  a1_start,
    int64_t *a1_stride,
    int64_t *a1_shape,
    int64_t  a1_ndim,

    int64_t axis
)
*/
int {{SYMBOL}}(int tool, ...)
{
    va_list list;                                   // **UNPACK PARAMETERS**
    va_start(list, tool);

    {{TYPE_A0}} *a0_first   = va_arg(list, {{TYPE_A0}}*);
    int64_t  a0_start   = va_arg(list, int64_t);    // Reduction result
    int64_t *a0_stride  = va_arg(list, int64_t*);
    int64_t *a0_shape   = va_arg(list, int64_t*);
    int64_t  a0_ndim    = va_arg(list, int64_t);

    {{TYPE_A1}} *a1_first    = va_arg(list, {{TYPE_A1}}*);
    int64_t  a1_start   = va_arg(list, int64_t);    // Input to reduce
    int64_t *a1_stride  = va_arg(list, int64_t*);
    int64_t *a1_shape   = va_arg(list, int64_t*);
    int64_t  a1_ndim    = va_arg(list, int64_t);

    int64_t axis = va_arg(list, int64_t);           // Reduction axis

    va_end(list);                                   // **DONE UNPACKING**

    {{TYPE_A0}} *a0_current = a0_first + a0_start;  // Point to first element in output.
    {{TYPE_A1}} *a1_current = a1_first + a1_start;  // Point to first element in input.
    {{TYPE_A1}} rvar = 0;                           // Use the first element as temp

    int64_t nelements = a1_shape[axis];
    int mthreads = omp_get_max_threads();
    int64_t nworkers = nelements > mthreads ? mthreads : 1;

    #pragma omp parallel for reduction(+:rvar) num_threads(nworkers)
    for(int64_t j=0; j<a1_shape[axis]; ++j) {
        {{TYPE_A1}} *tmp_current = a1_current + a1_stride[axis]*j;
        {{OPERATOR}};
    }
    *a0_current = rvar;
    
    return 1;
}

