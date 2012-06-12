/*
 * Copyright 2011 Simon A. F. Lund <safl@safl.dk>
 *
 * This file is part of cphVB.
 *
 * cphVB is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * cphVB is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with cphVB. If not, see <http://www.gnu.org/licenses/>.
 */
#include <cphvb.h>
#include <assert.h>
template <typename T0, typename T1, typename T2, typename Instr>
cphvb_error traverse_aaa( cphvb_instruction *instr, cphvb_index start, cphvb_index end ) {

    Instr opcode_func;                          // Element-wise functor-pointer

    cphvb_array *a0 = instr->operand[0];        // Operand pointers
    cphvb_array *a1 = instr->operand[1];
    cphvb_array *a2 = instr->operand[2];
                                                // Pointers to start of data elements
    T0* d0 = (T0*) cphvb_base_array(instr->operand[0])->data;
    T1* d1 = (T1*) cphvb_base_array(instr->operand[1])->data;
    T2* d2 = (T2*) cphvb_base_array(instr->operand[2])->data;

    assert(d0 != NULL);                         // Ensure that data is allocated
    assert(d1 != NULL);
    assert(d2 != NULL);

    cphvb_index j,                              // Traversal variables
                last_dim = a0->ndim-1,
                last_e  = (end>0) ? end : cphvb_nelements( a0->ndim, a0->shape )-1,
                cur_e;
    cphvb_index off0;                           // Stride-offset
    cphvb_index off1;
    cphvb_index off2;

    cphvb_index coord[CPHVB_MAXDIM];
    memset(coord, 0, CPHVB_MAXDIM * sizeof(cphvb_index));

    coord[last_dim] = start % a0->shape[last_dim];      // Fast-forward coordinates
    cur_e           = coord[last_dim];
    for(; cur_e < start; cur_e += a0->shape[last_dim] ) // As well as coordinates for the remaining dimensions
    {
        for(j = last_dim-1; (j >= 0) && (cur_e<start); --j )
        {
            coord[j]++;
            if (coord[j] < a0->shape[j]) {              // Still within this dimension
                break;
            } else {                                    // Reached the end of this dimension
                coord[j] = 0;                           // Reset coordinate
            }                                           // Loop then continues to increment the next dimension
        }
    }

    while( cur_e <= last_e )
    {
        off0 = a0->start;                           // Compute offset based on coord
        off1 = a1->start;
        off2 = a2->start;

        for( j=0; j<=last_dim; ++j)
        {
            off0 += coord[j] * a0->stride[j];
            off1 += coord[j] * a1->stride[j];
            off2 += coord[j] * a2->stride[j];
        }
                                                    // Iterate over "last" / "innermost" dimension
        for(; (coord[last_dim] < a0->shape[last_dim]) && (cur_e <= last_e); coord[last_dim]++, cur_e++ )    
        {
            opcode_func( (off0+d0), (off1+d1), (off2+d2) );

            off0 += a0->stride[last_dim];
            off1 += a1->stride[last_dim];
            off2 += a2->stride[last_dim];
        }
        coord[last_dim] = 0;

        for(j = last_dim-1; j >= 0; --j)            // Increment coordinates for the remaining dimensions
        {
            coord[j]++;
            if (coord[j] < a0->shape[j]) {          // Still within this dimension
                break;
            } else {                                // Reached the end of this dimension
                coord[j] = 0;                       // Reset coordinate
            }                                       // Loop then continues to increment the next dimension
        }

    }

    return CPHVB_SUCCESS;

}

template <typename T0, typename T1, typename T2, typename Instr>
cphvb_error traverse_aac( cphvb_instruction *instr, cphvb_index start, cphvb_index end ) {

    Instr opcode_func;                          // Element-wise functor-pointer

    cphvb_array *a0 = instr->operand[0];        // Operand pointers
    cphvb_array *a1 = instr->operand[1];
                                                // Pointers to start of data elements
    T0* d0 = (T0*) cphvb_base_array(instr->operand[0])->data;
    T1* d1 = (T1*) cphvb_base_array(instr->operand[1])->data;
    T2* d2 = (T2*) &(instr->constant.value);

    assert(d0 != NULL);                         // Ensure that data is allocated
    assert(d1 != NULL);

    cphvb_index j,                              // Traversal variables
                last_dim = a0->ndim-1,
                last_e  = (end>0) ? end : cphvb_nelements( a0->ndim, a0->shape )-1,
                cur_e;
    cphvb_index off0;                           // Stride-offset
    cphvb_index off1;

    cphvb_index coord[CPHVB_MAXDIM];
    memset(coord, 0, CPHVB_MAXDIM * sizeof(cphvb_index));

    coord[last_dim] = start % a0->shape[last_dim];      // Fast-forward coordinates
    cur_e           = coord[last_dim];
    for(; cur_e < start; cur_e += a0->shape[last_dim] ) // As well as coordinates for the remaining dimensions
    {
        for(j = last_dim-1; (j >= 0) && (cur_e<start); --j )
        {
            coord[j]++;
            if (coord[j] < a0->shape[j]) {              // Still within this dimension
                break;
            } else {                                    // Reached the end of this dimension
                coord[j] = 0;                           // Reset coordinate
            }                                           // Loop then continues to increment the next dimension
        }
    }

    while( cur_e <= last_e )
    {
        off0 = a0->start;                           // Compute offset based on coord
        off1 = a1->start;

        for( j=0; j<=last_dim; ++j)
        {
            off0 += coord[j] * a0->stride[j];
            off1 += coord[j] * a1->stride[j];
        }
                                                    // Iterate over "last" / "innermost" dimension
        for(; (coord[last_dim] < a0->shape[last_dim]) && (cur_e <= last_e); coord[last_dim]++, cur_e++ )    
        {
            opcode_func( (off0+d0), (off1+d1), d2 );

            off0 += a0->stride[last_dim];
            off1 += a1->stride[last_dim];
        }
        coord[last_dim] = 0;

        for(j = last_dim-1; j >= 0; --j)            // Increment coordinates for the remaining dimensions
        {
            coord[j]++;
            if (coord[j] < a0->shape[j]) {          // Still within this dimension
                break;
            } else {                                // Reached the end of this dimension
                coord[j] = 0;                       // Reset coordinate
            }                                       // Loop then continues to increment the next dimension
        }

    }

    return CPHVB_SUCCESS;

}

template <typename T0, typename T1, typename T2, typename Instr>
cphvb_error traverse_aca( cphvb_instruction *instr, cphvb_index start, cphvb_index end ) {

    Instr opcode_func;                          // Element-wise functor-pointer

    cphvb_array *a0 = instr->operand[0];        // Operand pointers
    cphvb_array *a2 = instr->operand[2];
                                                // Pointers to start of data elements
    T0* d0 = (T0*) cphvb_base_array(instr->operand[0])->data;
    T1* d1 = (T1*) &(instr->constant.value);
    T2* d2 = (T2*) cphvb_base_array(instr->operand[2])->data;

    assert(d0 != NULL);                         // Ensure that data is allocated
    assert(d2 != NULL);

    cphvb_index j,                              // Traversal variables
                last_dim = a0->ndim-1,
                last_e  = (end>0) ? end : cphvb_nelements( a0->ndim, a0->shape )-1,
                cur_e;
    cphvb_index off0;                           // Stride-offset
    cphvb_index off2;

    cphvb_index coord[CPHVB_MAXDIM];
    memset(coord, 0, CPHVB_MAXDIM * sizeof(cphvb_index));

    coord[last_dim] = start % a0->shape[last_dim];      // Fast-forward coordinates
    cur_e           = coord[last_dim];
    for(; cur_e < start; cur_e += a0->shape[last_dim] ) // As well as coordinates for the remaining dimensions
    {
        for(j = last_dim-1; (j >= 0) && (cur_e<start); --j )
        {
            coord[j]++;
            if (coord[j] < a0->shape[j]) {              // Still within this dimension
                break;
            } else {                                    // Reached the end of this dimension
                coord[j] = 0;                           // Reset coordinate
            }                                           // Loop then continues to increment the next dimension
        }
    }

    while( cur_e <= last_e )
    {
        off0 = a0->start;                           // Compute offset based on coord
        off2 = a2->start;

        for( j=0; j<=last_dim; ++j)
        {
            off0 += coord[j] * a0->stride[j];
            off2 += coord[j] * a2->stride[j];
        }
                                                    // Iterate over "last" / "innermost" dimension
        for(; (coord[last_dim] < a0->shape[last_dim]) && (cur_e <= last_e); coord[last_dim]++, cur_e++ )    
        {
            opcode_func( (off0+d0), d1, (off2+d2) );

            off0 += a0->stride[last_dim];
            off2 += a2->stride[last_dim];
        }
        coord[last_dim] = 0;

        for(j = last_dim-1; j >= 0; --j)            // Increment coordinates for the remaining dimensions
        {
            coord[j]++;
            if (coord[j] < a0->shape[j]) {          // Still within this dimension
                break;
            } else {                                // Reached the end of this dimension
                coord[j] = 0;                       // Reset coordinate
            }                                       // Loop then continues to increment the next dimension
        }

    }

    return CPHVB_SUCCESS;

}

template <typename T0, typename T1, typename Instr>
cphvb_error traverse_aa( cphvb_instruction *instr, cphvb_index start, cphvb_index end ) {

    Instr opcode_func;                          // Element-wise functor-pointer

    cphvb_array *a0 = instr->operand[0];        // Operand pointers
    cphvb_array *a1 = instr->operand[1];
                                                // Pointers to start of data elements
    T0* d0 = (T0*) cphvb_base_array(instr->operand[0])->data;
    T1* d1 = (T1*) cphvb_base_array(instr->operand[1])->data;

    assert(d0 != NULL);                         // Ensure that data is allocated
    assert(d1 != NULL);

    cphvb_index j,                              // Traversal variables
                last_dim = a0->ndim-1,
                last_e  = (end>0) ? end : cphvb_nelements( a0->ndim, a0->shape )-1,
                cur_e;
    cphvb_index off0;                           // Stride-offset
    cphvb_index off1;

    cphvb_index coord[CPHVB_MAXDIM];
    memset(coord, 0, CPHVB_MAXDIM * sizeof(cphvb_index));

    coord[last_dim] = start % a0->shape[last_dim];      // Fast-forward coordinates
    cur_e           = coord[last_dim];
    for(; cur_e < start; cur_e += a0->shape[last_dim] ) // As well as coordinates for the remaining dimensions
    {
        for(j = last_dim-1; (j >= 0) && (cur_e<start); --j )
        {
            coord[j]++;
            if (coord[j] < a0->shape[j]) {              // Still within this dimension
                break;
            } else {                                    // Reached the end of this dimension
                coord[j] = 0;                           // Reset coordinate
            }                                           // Loop then continues to increment the next dimension
        }
    }

    while( cur_e <= last_e )
    {
        off0 = a0->start;                           // Compute offset based on coord
        off1 = a1->start;

        for( j=0; j<=last_dim; ++j)
        {
            off0 += coord[j] * a0->stride[j];
            off1 += coord[j] * a1->stride[j];
        }
                                                    // Iterate over "last" / "innermost" dimension
        for(; (coord[last_dim] < a0->shape[last_dim]) && (cur_e <= last_e); coord[last_dim]++, cur_e++ )    
        {
            opcode_func( (off0+d0), (off1+d1) );

            off0 += a0->stride[last_dim];
            off1 += a1->stride[last_dim];
        }
        coord[last_dim] = 0;

        for(j = last_dim-1; j >= 0; --j)            // Increment coordinates for the remaining dimensions
        {
            coord[j]++;
            if (coord[j] < a0->shape[j]) {          // Still within this dimension
                break;
            } else {                                // Reached the end of this dimension
                coord[j] = 0;                       // Reset coordinate
            }                                       // Loop then continues to increment the next dimension
        }

    }

    return CPHVB_SUCCESS;

}

template <typename T0, typename T1, typename Instr>
cphvb_error traverse_ac( cphvb_instruction *instr, cphvb_index start, cphvb_index end ) {

    Instr opcode_func;                          // Element-wise functor-pointer

    cphvb_array *a0 = instr->operand[0];        // Operand pointers
                                                // Pointers to start of data elements
    T0* d0 = (T0*) cphvb_base_array(instr->operand[0])->data;
    T1* d1 = (T1*) &(instr->constant.value);

    assert(d0 != NULL);                         // Ensure that data is allocated

    cphvb_index j,                              // Traversal variables
                last_dim = a0->ndim-1,
                last_e  = (end>0) ? end : cphvb_nelements( a0->ndim, a0->shape )-1,
                cur_e;
    cphvb_index off0;                           // Stride-offset

    cphvb_index coord[CPHVB_MAXDIM];
    memset(coord, 0, CPHVB_MAXDIM * sizeof(cphvb_index));

    coord[last_dim] = start % a0->shape[last_dim];      // Fast-forward coordinates
    cur_e           = coord[last_dim];
    for(; cur_e < start; cur_e += a0->shape[last_dim] ) // As well as coordinates for the remaining dimensions
    {
        for(j = last_dim-1; (j >= 0) && (cur_e<start); --j )
        {
            coord[j]++;
            if (coord[j] < a0->shape[j]) {              // Still within this dimension
                break;
            } else {                                    // Reached the end of this dimension
                coord[j] = 0;                           // Reset coordinate
            }                                           // Loop then continues to increment the next dimension
        }
    }

    while( cur_e <= last_e )
    {
        off0 = a0->start;                           // Compute offset based on coord

        for( j=0; j<=last_dim; ++j)
        {
            off0 += coord[j] * a0->stride[j];
        }
                                                    // Iterate over "last" / "innermost" dimension
        for(; (coord[last_dim] < a0->shape[last_dim]) && (cur_e <= last_e); coord[last_dim]++, cur_e++ )    
        {
            opcode_func( (off0+d0), d1 );

            off0 += a0->stride[last_dim];
        }
        coord[last_dim] = 0;

        for(j = last_dim-1; j >= 0; --j)            // Increment coordinates for the remaining dimensions
        {
            coord[j]++;
            if (coord[j] < a0->shape[j]) {          // Still within this dimension
                break;
            } else {                                // Reached the end of this dimension
                coord[j] = 0;                       // Reset coordinate
            }                                       // Loop then continues to increment the next dimension
        }

    }

    return CPHVB_SUCCESS;

}



