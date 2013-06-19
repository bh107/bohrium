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

void {{SYMBOL}}(int tool, ...)
{
    va_list list;               // Unpack arguments
    va_start(list, tool);

    *a0_current = va_arg(list, {{TYPE_A0}}*);
 
    {{#a1_dense}}
    {{TYPE_A1}} *a1_current   = va_arg(list, {{TYPE_A1}}*);
    {{TYPE_A1}} *a1_first     = a1_current;
    int64_t  a1_start   = va_arg(list, int64_t);
    int64_t *a1_stride  = va_arg(list, int64_t*);
    assert(a1_current != NULL);
    {{/a1_dense}}

    {{#a2_dense}}
    {{TYPE_A2}} *a2_current   = va_arg(list, {{TYPE_A2}}*);
    {{TYPE_A2}} *a2_first     = a2_current;
    int64_t  a2_start   = va_arg(list, int64_t);
    int64_t *a2_stride  = va_arg(list, int64_t*);
    assert(a2_current != NULL);
    {{/a2_dense}}

    {{#a3_dense}}
    {{TYPE_A3}} *a3_current   = va_arg(list, {{TYPE_A3}}*);
    {{TYPE_A3}} *a3_first     = a3_current;
    int64_t  a3_start   = va_arg(list, int64_t);
    int64_t *a3_stride  = va_arg(list, int64_t*);
    assert(a3_current != NULL);
    {{/a3_dense}}

    {{#a4_dense}}
    {{TYPE_A4}} *a4_current   = va_arg(list, {{TYPE_A4}}*);
    {{TYPE_A4}} *a4_first     = a4_current;
    int64_t  a4_start   = va_arg(list, int64_t);
    int64_t *a4_stride  = va_arg(list, int64_t*);
    assert(a4_current != NULL);
    {{/a4_dense}}

    {{#a5_dense}}
    {{TYPE_A5}} *a5_current   = va_arg(list, {{TYPE_A5}}*);
    {{TYPE_A5}} *a5_first     = a5_current;
    int64_t  a5_start   = va_arg(list, int64_t);
    int64_t *a5_stride  = va_arg(list, int64_t*);
    assert(a5_current != NULL);
    {{/a5_dense}}

    {{#a6_dense}}
    {{TYPE_A6}} *a6_current   = va_arg(list, {{TYPE_A6}}*);
    {{TYPE_A6}} *a6_first     = a6_current;
    int64_t  a6_start   = va_arg(list, int64_t);
    int64_t *a6_stride  = va_arg(list, int64_t*);
    assert(a6_current != NULL);
    {{/a6_dense}}

    {{#a7_dense}}
    {{TYPE_A7}} *a7_current   = va_arg(list, {{TYPE_A7}}*);
    {{TYPE_A7}} *a7_first     = a7_current;
    int64_t  a7_start   = va_arg(list, int64_t);
    int64_t *a7_stride  = va_arg(list, int64_t*);
    assert(a7_current != NULL);
    {{/a7_dense}}
    
    int64_t *shape      = va_arg(list, int64_t*);
    int64_t ndim        = va_arg(list, int64_t);
    int64_t nelements   = va_arg(list, int64_t);

    va_end(list);

    int64_t j,                  // Traversal variables
            last_dim    = ndim-1,
            last_e      = nelements-1;

    int64_t cur_e = 0;
    int64_t shape_ld = shape[last_dim];

    {{#a1_dense}}int64_t a1_stride_ld = a1_stride[last_dim];{{/a1_dense}}
    {{#a2_dense}}int64_t a2_stride_ld = a2_stride[last_dim];{{/a2_dense}}
    {{#a3_dense}}int64_t a3_stride_ld = a3_stride[last_dim];{{/a3_dense}}
    {{#a4_dense}}int64_t a4_stride_ld = a4_stride[last_dim];{{/a4_dense}}
    {{#a5_dense}}int64_t a5_stride_ld = a5_stride[last_dim];{{/a5_dense}}
    {{#a6_dense}}int64_t a6_stride_ld = a6_stride[last_dim];{{/a6_dense}}
    {{#a7_dense}}int64_t a7_stride_ld = a7_stride[last_dim];{{/a7_dense}}

    int64_t coord[DYNAMITE_MAXDIM];
    memset(coord, 0, DYNAMITE_MAXDIM * sizeof(int64_t));

    // Start by assigning the neutral element to the output
    *a0_current = 0;    // probably not the case for all reductions..

    while (cur_e <= last_e) {
        
        //a0_current = a0_first + a0_start;         // Reset offsets
        //a0_current remains untouched...
        {{#a1_dense}}a1_current = a1_first + a1_start;{{/a1_dense}}
        {{#a2_dense}}a2_current = a2_first + a2_start;{{/a2_dense}}
        {{#a3_dense}}a3_current = a3_first + a3_start;{{/a3_dense}}
        {{#a4_dense}}a4_current = a4_first + a4_start;{{/a4_dense}}
        {{#a5_dense}}a5_current = a5_first + a5_start;{{/a5_dense}}
        {{#a6_dense}}a6_current = a6_first + a6_start;{{/a6_dense}}
        {{#a7_dense}}a7_current = a7_first + a7_start;{{/a7_dense}}
        for (j=0; j<=last_dim; ++j) {           // Compute offset based on coordinate
            //a0_current += coord[j] * a0_stride[j];
            //a0_current remains untouched
            {{#a1_dense}}a1_current += coord[j] * a1_stride[j];{{/a1_dense}}
            {{#a2_dense}}a2_current += coord[j] * a2_stride[j];{{/a2_dense}}
            {{#a3_dense}}a3_current += coord[j] * a3_stride[j];{{/a3_dense}}
            {{#a4_dense}}a4_current += coord[j] * a4_stride[j];{{/a4_dense}}
            {{#a5_dense}}a5_current += coord[j] * a5_stride[j];{{/a5_dense}}
            {{#a6_dense}}a6_current += coord[j] * a6_stride[j];{{/a6_dense}}
            {{#a7_dense}}a7_current += coord[j] * a7_stride[j];{{/a7_dense}}
        }

        for (j = 0; j < shape_ld; j++) {        // Iterate over "last" / "innermost" dimension
            {{OPERATOR}};

            // *a0_current += (float)( sqrt(((*a1_current)*(*a2_current))+((*a3_current)*(*a4_current))) <=(*a5_current));

            //a0_current += a0_stride_ld;
            // a0_current += a0_stride_ld; remains untouched
            {{#a1_dense}}a1_current += a1_stride_ld;{{/a1_dense}}
            {{#a2_dense}}a2_current += a2_stride_ld;{{/a2_dense}}
            {{#a3_dense}}a3_current += a3_stride_ld;{{/a3_dense}}
            {{#a4_dense}}a4_current += a4_stride_ld;{{/a4_dense}}
            {{#a5_dense}}a5_current += a5_stride_ld;{{/a5_dense}}
            {{#a6_dense}}a6_current += a6_stride_ld;{{/a6_dense}}
            {{#a7_dense}}a7_current += a7_stride_ld;{{/a7_dense}}
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

