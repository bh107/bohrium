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
#include "stdlib.h"
#include "stdarg.h"
#include "string.h"
#include "assert.h"

#define DYNAMITE_MAXDIM 16

/*
void traverse_aaa(int64_t a0_start, int64_t* a0_stride, {{TYPE}}* a0_data,
              int64_t a1_start, int64_t* a1_stride, {{TYPE}}* a1_data,
              int64_t a2_start, int64_t* a2_stride, {{TYPE}}* a2_data,
              int64_t* shape,
              int64_t ndim,
              int64_t nelements)
*/
void traverse_aaa(int tool, ...)
{
    va_list list;
    va_start(list, tool);

    int64_t a0_start    = va_arg(list, int64_t);
    int64_t* a0_stride  = va_arg(list, int64_t*);
    {{TYPE}}* a0_data   = va_arg(list, {{TYPE}}*);

    int64_t a1_start    = va_arg(list, int64_t);
    int64_t* a1_stride  = va_arg(list, int64_t*);
    {{TYPE}}* a1_data   = va_arg(list, {{TYPE}}*);

    int64_t a2_start    = va_arg(list, int64_t);
    int64_t* a2_stride  = va_arg(list, int64_t*);
    {{TYPE}}* a2_data   = va_arg(list, {{TYPE}}*);
    
    int64_t* shape      = va_arg(list, int64_t*);
    int64_t ndim        = va_arg(list, int64_t);
    int64_t nelements   = va_arg(list, int64_t);

    va_end(list);

    assert(a0_data != NULL);    // Ensure that data is allocated
    assert(a1_data != NULL);
    assert(a2_data != NULL);

    int64_t j,                  // Traversal variables
            last_dim    = ndim-1,
            last_e      = nelements-1;

    int64_t coord[DYNAMITE_MAXDIM];
    int64_t cur_e = 0;

    int64_t off0;               // Stride-offset
    int64_t off1;
    int64_t off2;

    memset(coord, 0, DYNAMITE_MAXDIM * sizeof(int64_t));

    while (cur_e <= last_e) {
        off0 = a0_start;                       // Reset offset
        off1 = a1_start;
        off2 = a2_start;

        for (j=0; j<=last_dim; ++j) {           // Compute offset based on coordinate
            off0 += coord[j] * a0_stride[j];
            off1 += coord[j] * a1_stride[j];
            off2 += coord[j] * a2_stride[j];
        }
                                                // Iterate over "last" / "innermost" dimension
        for (; (coord[last_dim] < shape[last_dim]) && (cur_e <= last_e); coord[last_dim]++, cur_e++) {
            *(off0+a0_data) = *(off1+a1_data) {{OPERATOR}} *(off2+a2_data);

            off0 += a0_stride[last_dim];
            off1 += a1_stride[last_dim];
            off2 += a2_stride[last_dim];
        }

        if (coord[last_dim] >= shape[last_dim]) {
            coord[last_dim] = 0;
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
}

