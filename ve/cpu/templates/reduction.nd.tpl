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
    va_list list;                                   // **Unpack arguments**
    va_start(list, tool);

    {{TYPE_A0}} *a0_current;
    {{TYPE_A0}} *a0_first   = va_arg(list, {{TYPE_A0}}*);
    int64_t  a0_start   = va_arg(list, int64_t);    // Reduction result
    int64_t *a0_stride  = va_arg(list, int64_t*);
    int64_t *a0_shape   = va_arg(list, int64_t*);
    int64_t  a0_ndim    = va_arg(list, int64_t);

    //{{TYPE_A1}} *a1_current;
    {{TYPE_A1}} *a1_first    = va_arg(list, {{TYPE_A1}}*);
    int64_t  a1_start   = va_arg(list, int64_t);    // Input to reduce
    int64_t *a1_stride  = va_arg(list, int64_t*);
    int64_t *a1_shape   = va_arg(list, int64_t*);
    int64_t  a1_ndim    = va_arg(list, int64_t);

    int64_t axis = va_arg(list, int64_t);           // Reduction axis

    va_end(list);                                   // **DONE**

    int64_t a1_i;               // Iterator variables...

    {{TYPE_A1}} *tmp_current;    // Intermediate array
    {{TYPE_A1}} *tmp_first;      
    int64_t tmp_start;
    int64_t tmp_stride[CPU_MAXDIM];    

    if (1 == a1_ndim) {                             // ** 1D Special Case **
        a0_current = a0_first + a0_start;           // Point to first element in output.
        {{TYPE_A0}} rvar = *(a1_first+a1_start);    // Use the first element as temp
        for(tmp_current = a1_first+a1_start+a1_stride[axis], a1_i=1;
            a1_i < a1_shape[axis];
            tmp_current += a1_stride[axis], a1_i++) {
            
            {{OPERATOR}};
        }
        *a0_current = rvar;
        return 1;
    } else {                                    // ** ND General Case **
        int64_t j,                              // Traversal variables
                last_dim,
                last_e,
                cur_e,
                coord[CPU_MAXDIM];

        tmp_first    = a1_first;                  // Use the temporary as a copy of input
        tmp_start   = a1_start;                 // without the 'axis' dimension

        int64_t tmp_dim;
        for (tmp_dim=0, a1_i=0; a1_i<a1_ndim; ++a1_i) { // Excluding the 'axis' dimension.
            if (a1_i != axis) {
                tmp_stride[tmp_dim]   = a1_stride[a1_i];
                ++tmp_dim;
            }
        }

        last_e = 1;
        int64_t k;
        for (k = 0; k < a0_ndim; ++k) { // COUNT THE ELEMENTS
            last_e *= a0_shape[k];
        }
        --last_e;

        last_dim = a0_ndim-1;

        for(a1_i=0; a1_i<a1_shape[axis]; ++a1_i, tmp_start += a1_stride[axis]) {

            cur_e = 0;                                  // Reset coordinate and element counter
            memset(coord, 0, CPU_MAXDIM * sizeof(int64_t));

            while (cur_e <= last_e) {
                a0_current   = a0_first + a0_start;       // Reset offsets
                tmp_current  = tmp_first + tmp_start;

                for (j=0; j<=last_dim; ++j) {           // Compute offset based on coordinate
                    a0_current   += coord[j] * a0_stride[j];
                    tmp_current  += coord[j] * tmp_stride[j];
                }
                                                        // Iterate over "last" / "innermost" dimension
                if (0==a1_i) {                         // First off, copy the intermediate value
                    for(;
                        (coord[last_dim] < a0_shape[last_dim]) && (cur_e <= last_e);
                        a0_current   += a0_stride[last_dim], // Increment element indexes
                        tmp_current  += tmp_stride[last_dim],

                        coord[last_dim]++,                  // Increment coordinates
                        cur_e++
                    ) {
                        *a0_current = *tmp_current;
                    }
                } else {                                // Then do the actual reduction
                    for(;
                        (coord[last_dim] < a0_shape[last_dim]) && (cur_e <= last_e);

                        a0_current   += a0_stride[last_dim], // Offsets
                        tmp_current  += tmp_stride[last_dim],
                        coord[last_dim]++,                  // Coordinates
                        cur_e++
                    ) {
                        //{{OPERATOR}}; // TODO: FIX
                        *a0_current += *tmp_current;
                    }
                }

                if (coord[last_dim] >= a0_shape[last_dim]) {
                    coord[last_dim] = 0;
                    for(j = last_dim-1; j >= 0; --j) {  // Increment coordinates for the remaining dimensions
                        coord[j]++;
                        if (coord[j] < a0_shape[j]) {   // Still within this dimension
                            break;
                        } else {                        // Reached the end of this dimension
                            coord[j] = 0;               // Reset coordinate
                        }                               // Loop then continues to increment the next dimension
                    }
                }
            }
        }
        return 1;
    }
}

