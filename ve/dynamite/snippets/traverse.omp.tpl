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

void {{SYMBOL}}(int tool, ...)
{
    va_list list;               // Unpack arguments
    va_start(list, tool);

    int64_t  a0_start   = va_arg(list, int64_t);
    int64_t *a0_stride  = va_arg(list, int64_t*);
    {{TYPE_OUT}} *a0_data   = va_arg(list, {{TYPE_OUT}}*);
    {{TYPE_OUT}} *a0_offset;
    
    {{#a1_dense}}
    int64_t  a1_start   = va_arg(list, int64_t);
    int64_t *a1_stride  = va_arg(list, int64_t*);
    {{TYPE_IN1}} *a1_data   = va_arg(list, {{TYPE_IN1}}*);
    {{TYPE_IN1}} *a1_offset;
    {{/a1_dense}}

    {{#a1_scalar}}
    {{TYPE_IN1}} *a1_offset   = va_arg(list, {{TYPE_IN1}}*);
    {{/a1_scalar}}

    {{#a2_dense}}
    int64_t  a2_start   = va_arg(list, int64_t);
    int64_t *a2_stride  = va_arg(list, int64_t*);
    {{TYPE_IN2}} *a2_data   = va_arg(list, {{TYPE_IN2}}*);
    {{TYPE_IN2}} *a2_offset;
    {{/a2_dense}}

    {{#a2_scalar}}
    {{TYPE_IN2}} *a2_offset   = va_arg(list, {{TYPE_IN2}}*);
    {{/a2_scalar}}
    
    int64_t *shape      = va_arg(list, int64_t*);
    int64_t ndim        = va_arg(list, int64_t);
    int64_t nelements   = va_arg(list, int64_t);

    va_end(list);

    assert(a0_data != NULL);    // Ensure that data is allocated
    {{#a1_dense}}assert(a1_data != NULL);{{/a1_dense}}
    {{#a2_dense}}assert(a2_data != NULL);{{/a2_dense}}

    int64_t j,                  // Traversal variables
            last_dim    = ndim-1,
            last_e      = nelements-1;

    int64_t cur_e = 0;
    int64_t shape_ld = shape[last_dim];
    int64_t a0_stride_ld = a0_stride[last_dim];
    {{#a1_dense}}int64_t a1_stride_ld = a1_stride[last_dim];{{/a1_dense}}
    {{#a2_dense}}int64_t a2_stride_ld = a2_stride[last_dim];{{/a2_dense}}

    int64_t coord[DYNAMITE_MAXDIM];
    memset(coord, 0, DYNAMITE_MAXDIM * sizeof(int64_t));

    while (cur_e <= last_e) {
        
        a0_offset = a0_data + a0_start;         // Reset offsets
        {{#a1_dense}}a1_offset = a1_data + a1_start;{{/a1_dense}}
        {{#a2_dense}}a2_offset = a2_data + a2_start;{{/a2_dense}}
        for (j=0; j<=last_dim; ++j) {           // Compute offset based on coordinate
            a0_offset += coord[j] * a0_stride[j];
            {{#a1_dense}}a1_offset += coord[j] * a1_stride[j];{{/a1_dense}}
            {{#a2_dense}}a2_offset += coord[j] * a2_stride[j];{{/a2_dense}}
        }

        {{TYPE_OUT}} *a0_offset_old;
        {{#a1_dense}}{{TYPE_IN1}} *a1_offset_old;{{/a1_dense}}
        {{#a2_dense}}{{TYPE_IN2}} *a2_offset_old;{{/a2_dense}}

        a0_offset_old = a0_offset;
        {{#a1_dense}}a1_offset_old = a1_offset;{{/a1_dense}}
        {{#a2_dense}}a2_offset_old = a2_offset;{{/a2_dense}}

        #pragma omp parallel for private(j, a0_offset {{#a1_dense}},a1_offset{{/a1_dense}} {{#a2_dense}},a2_offset{{/a2_dense}})
        for (j = 0; j < shape_ld; j++) {        // Iterate over "last" / "innermost" dimension
            a0_offset = a0_offset_old + a0_stride_ld *j;
            {{#a1_dense}}a1_offset = a1_offset_old + a1_stride_ld *j;{{/a1_dense}}
            {{#a2_dense}}a2_offset = a2_offset_old + a2_stride_ld *j;{{/a2_dense}}

            {{OPERATOR}};
        }
        cur_e += shape_ld;

        a0_offset = a0_offset_old+a0_stride_ld*(shape_ld-1);
        {{#a1_dense}}a1_offset = a1_offset_old+a1_stride_ld*(shape_ld-1);{{/a1_dense}}
        {{#a2_dense}}a2_offset = a2_offset_old+a2_stride_ld*(shape_ld-1);{{/a2_dense}}

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

