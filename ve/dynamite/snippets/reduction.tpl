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

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define DEG_CIR 360.0
#define DEG_RAD (M_PI / (DEG_CIR / 2.0))
#define RAD_DEG ((DEG_CIR / 2.0) / M_PI)

#ifndef DYNAMITE_MISC
#define DYNAMITE_MAXDIM 16
#endif

/*
int reduction(
    int tool,
    int64_t a0_start, int64_t *a0_stride, T *a0_data,
    int64_t a1_start, int64_t *a0_stride, T *a1_data,
    int64_t *a1_shape,
    int64_t a1_ndim,
    int64_t axis
)
*/
int {{SYMBOL}}(int tool, ...)
{
    va_list list;                                   // **Unpack arguments**
    va_start(list, tool);

    {{TYPE_A0}} *a0_data   = va_arg(list, {{TYPE_A0}}*);
    int64_t  a0_start   = va_arg(list, int64_t);    // Reduction result
    int64_t *a0_stride  = va_arg(list, int64_t*);
    {{TYPE_A0}} *a0_offset;

    {{TYPE_A1}} *a1_data    = va_arg(list, {{TYPE_A1}}*);
    int64_t  a1_start   = va_arg(list, int64_t);    // Input to reduce
    int64_t *a1_stride  = va_arg(list, int64_t*);
    {{TYPE_A1}} *a1_offset;

    int64_t *a1_shape  = va_arg(list, int64_t*);    // Shape of the input
    int64_t a1_ndim    = va_arg(list, int64_t);     // Number of dimensions in input

    int64_t axis = va_arg(list, int64_t);           // Reduction axis

    va_end(list);                                   // **DONE**

    int64_t i, j;   // Iterator variables...

    if (1 == a1_ndim) {                         // ** 1D Special Case **
        a0_offset = a0_data + a0_start;         // Point to first element in output.
        *a0_offset = *(a1_data+a1_start);       // Use the first element as temp
        for(a1_offset = a1_data+a1_start+a1_stride[axis], j=1;
            j < a1_shape[axis];
            a1_offset += a1_stride[axis], j++) {
            
            // {{OPERATOR}}
            *a0_offset = *a0_offset + *a1_offset; // TODO: Inline cexpr here
        }
        return 1;
    } else {                                // ** ND General Case **

        int64_t tmp_start = a1_start;       // Intermediate results
        int64_t tmp_stride[DYNAMITE_MAXDIM];
        {{TYPE_A1}} *tmp_data;

        int64_t tmp_shape[DYNAMITE_MAXDIM]; // 
        int64_t tmp_ndim = a1_ndim-1;          // Reduce dimensionality

        for (j=0, i=0; i<a1_ndim; ++i) {        // Copy every dimension except for the 'axis' dimension.
            if (i != axis) {
                tmp_shape[j]    = a1_shape[i];
                tmp_stride[j]   = a1_stride[i];
                ++j;
            }
        }
        tmp_data = a1_data;                 // Set data / base pointer

        // INSERT TRAVERSE CODE HERE FOR COPYING THE FIRST ELEMENT USING IDENTITY
        // traverse(out, tmp, IDENTITY)
        // BH_IDENTITY_DD(a0, tmp)

        tmp_start += a1_stride[axis];

        for(i=1; i<a1_shape[axis]; ++i) {
            // INSERT TRAVERSE CODE HERE FOR DOING THE ACTUAL REDUCTION (ADD/MUL/WHATEVER)
            // BH_ADD_DDD(a0, a0, tmp);
            tmp_start += a1_stride[axis];
        }

        return 1;
    }

}

