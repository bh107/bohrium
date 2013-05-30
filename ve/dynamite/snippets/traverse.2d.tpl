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

    {{TYPE_OUT}} *a0_current = va_arg(list, {{TYPE_OUT}}*);
    int64_t  a0_start   = va_arg(list, int64_t);
    int64_t *a0_stride  = va_arg(list, int64_t*);
    assert(a0_current != NULL);

    {{#a1_scalar}}
    {{TYPE_IN1}} *a1_current   = va_arg(list, {{TYPE_IN1}}*);
    {{/a1_scalar}}  
 
    {{#a1_dense}}
    {{TYPE_IN1}} *a1_current   = va_arg(list, {{TYPE_IN1}}*);
    int64_t  a1_start   = va_arg(list, int64_t);
    int64_t *a1_stride  = va_arg(list, int64_t*);
    assert(a1_current != NULL);
    {{/a1_dense}}

    {{#a2_scalar}}
    {{TYPE_IN2}} *a2_current   = va_arg(list, {{TYPE_IN2}}*);
    {{/a2_scalar}}

    {{#a2_dense}}
    {{TYPE_IN2}} *a2_current   = va_arg(list, {{TYPE_IN2}}*);
    int64_t  a2_start   = va_arg(list, int64_t);
    int64_t *a2_stride  = va_arg(list, int64_t*);
    assert(a2_current != NULL);
    {{/a2_dense}}
    
    int64_t *shape      = va_arg(list, int64_t*);
    int64_t ndim        = va_arg(list, int64_t);
    va_end(list);

    int64_t i, j,                  // Traversal variables
            ld  = ndim-1,
            nld = ld-1;

    int64_t a0_stride_ld    = a0_stride[ld];
    int64_t a0_stride_nld   = a0_stride[nld];
    a0_current += a0_start;

    int64_t a0_rewind_ld = shape[ld]*a0_stride[ld];

    {{#a1_dense}}
    int64_t a1_rewind_ld    = shape[ld]*a1_stride[ld];
    int64_t a1_stride_ld    = a1_stride[ld];
    int64_t a1_stride_nld   = a1_stride[nld];
    a1_current += a1_start;
    {{/a1_dense}}

    {{#a2_dense}}
    int64_t a2_rewind_ld    = shape[ld]*a2_stride[ld];
    int64_t a2_stride_ld    = a2_stride[ld];
    int64_t a2_stride_nld   = a2_stride[nld];
    a2_current += a2_start;
    {{/a2_dense}}

    for (j = 0; j < shape[nld]; ++j) {
        for (i = 0; i < shape[ld]; ++i) {
            {{OPERATOR}};

            a0_current += a0_stride_ld;
            {{#a1_dense}}a1_current += a1_stride_ld;{{/a1_dense}}
            {{#a2_dense}}a2_current += a2_stride_ld;{{/a2_dense}}
        }
        a0_current -= a0_rewind_ld;
        a0_current += a0_stride_nld;
        {{#a1_dense}}
        a1_current -= a1_rewind_ld;
        a1_current += a1_stride_nld;
        {{/a1_dense}}
        {{#a2_dense}}
        a2_current -= a2_rewind_ld;
        a2_current += a2_stride_nld;
        {{/a2_dense}}
    }

}

