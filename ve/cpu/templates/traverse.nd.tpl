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

#ifndef CPU_MISC
#define CPU_MAXDIM 16
#endif
{{/include}}

void {{SYMBOL}}(int tool, ...)
{
    va_list list;               // Unpack arguments
    va_start(list, tool);

    {{TYPE_A0}} *a0_current = va_arg(list, {{TYPE_A0}}*);
    {{TYPE_A0}} *a0_first = a0_current;
    int64_t  a0_start   = va_arg(list, int64_t);
    int64_t *a0_stride  = va_arg(list, int64_t*);
    assert(a0_current != NULL);

    {{#a1_scalar}}
    {{TYPE_A1}} *a1_current   = va_arg(list, {{TYPE_A1}}*);
    {{/a1_scalar}}  
 
    {{#a1_dense}}
    {{TYPE_A1}} *a1_current   = va_arg(list, {{TYPE_A1}}*);
    {{TYPE_A1}} *a1_first     = a1_current;
    int64_t  a1_start   = va_arg(list, int64_t);
    int64_t *a1_stride  = va_arg(list, int64_t*);
    assert(a1_current != NULL);
    {{/a1_dense}}

    {{#a2_scalar}}
    {{TYPE_A2}} *a2_current   = va_arg(list, {{TYPE_A2}}*);
    {{/a2_scalar}}

    {{#a2_dense}}
    {{TYPE_A2}} *a2_current   = va_arg(list, {{TYPE_A2}}*);
    {{TYPE_A2}} *a2_first     = a2_current;
    int64_t  a2_start   = va_arg(list, int64_t);
    int64_t *a2_stride  = va_arg(list, int64_t*);
    assert(a2_current != NULL);
    {{/a2_dense}}
    
    int64_t *shape      = va_arg(list, int64_t*);
    int64_t ndim        = va_arg(list, int64_t);

    va_end(list);

    int64_t nelements = 1;      // Compute number of elements
    int k;
    for (k = 0; k<ndim; ++k){
        nelements *= shape[k];
    }

    int64_t j,                  // Traversal variables
            last_dim    = ndim-1,
            last_e      = nelements-1;

    int64_t cur_e = 0;
    int64_t shape_ld = shape[last_dim];
    int64_t a0_stride_ld = a0_stride[last_dim];
    {{#a1_dense}}int64_t a1_stride_ld = a1_stride[last_dim];{{/a1_dense}}
    {{#a2_dense}}int64_t a2_stride_ld = a2_stride[last_dim];{{/a2_dense}}

    int64_t coord[CPU_MAXDIM];
    memset(coord, 0, CPU_MAXDIM * sizeof(int64_t));

    while (cur_e <= last_e) {
        
        a0_current = a0_first + a0_start;         // Reset offsets
        {{#a1_dense}}a1_current = a1_first + a1_start;{{/a1_dense}}
        {{#a2_dense}}a2_current = a2_first + a2_start;{{/a2_dense}}
        for (j=0; j<=last_dim; ++j) {           // Compute offset based on coordinate
            a0_current += coord[j] * a0_stride[j];
            {{#a1_dense}}a1_current += coord[j] * a1_stride[j];{{/a1_dense}}
            {{#a2_dense}}a2_current += coord[j] * a2_stride[j];{{/a2_dense}}
        }

        for (j = 0; j < shape_ld; j++) {        // Iterate over "last" / "innermost" dimension
            {{OPERATOR}};

            a0_current += a0_stride_ld;
            {{#a1_dense}}a1_current += a1_stride_ld;{{/a1_dense}}
            {{#a2_dense}}a2_current += a2_stride_ld;{{/a2_dense}}
        }
        cur_e += shape_ld;

        // coord[last_dim] is never used, only all the other coord[dim!=last_dim]
        for(j = last_dim-1; j >= 0; --j) {  // Increment coordinates for the remaining dimensions
            coord[j]++;
            if (coord[j] < shape[j]) {      // Still within this dimension
                break;
            } else {                        // Reached the end of this dimension
                coord[j] = 0;               // Reset coordinate
            }                               // Loop then continues to increment the next dimension
        }
    }
}

