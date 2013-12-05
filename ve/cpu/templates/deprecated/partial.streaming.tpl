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

// TODO: Deal with this differently!
#define BH_MAXDIM (16)

typedef struct bh_array bh_array;
struct bh_array
{
    /// Pointer to the base array. If NULL this is a base array
    bh_array*     base;

    /// The type of data in the array
    int64_t type;

    /// Number of dimentions
    int64_t ndim;

    /// Index of the start element (always 0 for base-array)
    int64_t start;

    /// Number of elements in each dimention
    int64_t shape[BH_MAXDIM];

    /// The stride for each dimention
    int64_t stride[BH_MAXDIM];

    /// Pointer to the actual data. Ignored for views
    void *data;
};
// TODO: Deal with this differently!

void {{SYMBOL}}(size_t noperands, bh_array **operands)
{
    {{TYPE_A0}} *a0_current = ({{TYPE_A0}}*)operands[0]->data;
    {{TYPE_A0}} *a0_first   = a0_current;
    int64_t  a0_start   = operands[1]->start;
    int64_t *a0_stride  = operands[1]->stride;
    assert(a0_current != NULL);
 
    {{#a1_dense}}
    {{TYPE_A1}} *a1_current = ({{TYPE_A1}}*)operands[1]->data;
    {{TYPE_A1}} *a1_first   = a1_current;
    int64_t  a1_start   = operands[1]->start;
    int64_t *a1_stride  = operands[1]->stride;
    assert(a1_current != NULL);
    {{/a1_dense}}

    {{#a2_dense}}
    {{TYPE_A2}} *a2_current   = ({{TYPE_A2}}*)operands[2]->data;
    {{TYPE_A2}} *a2_first     = a2_current;
    int64_t  a2_start   = operands[2]->start;
    int64_t *a2_stride  = operands[2]->stride;
    assert(a2_current != NULL);
    {{/a2_dense}}

    {{#a3_dense}}
    {{TYPE_A3}} *a3_current   = ({{TYPE_A3}}*)operands[3]->data;
    {{TYPE_A3}} *a3_first     = a3_current;
    int64_t  a3_start   = operands[3]->start;
    int64_t *a3_stride  = operands[3]->stride;
    assert(a3_current != NULL);
    {{/a3_dense}}

    {{#a4_dense}}
    {{TYPE_A4}} *a4_current   = ({{TYPE_A4}}*)operands[4]->data;
    {{TYPE_A4}} *a4_first     = a4_current;
    int64_t  a4_start   = operands[4]->start;
    int64_t *a4_stride  = operands[4]->stride;
    assert(a4_current != NULL);
    {{/a4_dense}}

    {{#a5_dense}}
    {{TYPE_A5}} *a5_current   = ({{TYPE_A5}}*)operands[5]->data;
    {{TYPE_A5}} *a5_first     = a5_current;
    int64_t  a5_start   = operands[5]->start;
    int64_t *a5_stride  = operands[5]->stride;
    assert(a5_current != NULL);
    {{/a5_dense}}

    {{#a6_dense}}
    {{TYPE_A6}} *a6_current   = ({{TYPE_A6}}*)operands[6]->data;
    {{TYPE_A6}} *a6_first     = a6_current;
    int64_t  a6_start   = operands[6]->start;
    int64_t *a6_stride  = operands[6]->stride;
    assert(a6_current != NULL);
    {{/a6_dense}}

    {{#a7_dense}}
    {{TYPE_A7}} *a7_current   = ({{TYPE_A7}}*)operands[7]->data;
    {{TYPE_A7}} *a7_first     = a7_current;
    int64_t  a7_start   = operands[7]->start;
    int64_t *a7_stride  = operands[7]->stride;
    assert(a7_current != NULL);
    {{/a7_dense}}

    int64_t nelements = 1;
    for (int64_t i = 0; i < operands[1]->ndim; ++i) {
        nelements *= operands[1]->shape[i];
    }

    int64_t j,                  // Traversal variables
            last_dim    = operands[1]->ndim-1,
            last_e      = nelements-1;

    int64_t cur_e = 0;
    int64_t shape_ld = operands[1]->shape[last_dim];

    {{#a1_dense}}int64_t a1_stride_ld = a1_stride[last_dim];{{/a1_dense}}
    {{#a2_dense}}int64_t a2_stride_ld = a2_stride[last_dim];{{/a2_dense}}
    {{#a3_dense}}int64_t a3_stride_ld = a3_stride[last_dim];{{/a3_dense}}
    {{#a4_dense}}int64_t a4_stride_ld = a4_stride[last_dim];{{/a4_dense}}
    {{#a5_dense}}int64_t a5_stride_ld = a5_stride[last_dim];{{/a5_dense}}
    {{#a6_dense}}int64_t a6_stride_ld = a6_stride[last_dim];{{/a6_dense}}
    {{#a7_dense}}int64_t a7_stride_ld = a7_stride[last_dim];{{/a7_dense}}

    int64_t coord[CPU_MAXDIM];
    memset(coord, 0, CPU_MAXDIM * sizeof(int64_t));

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
            if (coord[j] < operands[1]->shape[j]) {      // Still within this dimension
                break;
            } else {                        // Reached the end of this dimension
                coord[j] = 0;               // Reset coordinate
            }                               // Loop then continues to increment the next dimension
        }
    }
}

