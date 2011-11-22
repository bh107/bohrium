#include <cstring>
#include <iostream>
#include <cphvb.h>

cphvb_intp const_stride[CPHVB_MAXDIM] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

template <typename T, typename Instr>
cphvb_error traverse_3( cphvb_instruction *instr ) {

    Instr opcode_func;

    T *d0, *d1, *d2;                            // Pointers to start of data elements
    cphvb_array *a0 = instr->operand[0],        // Operands
                *a1 = instr->operand[1],
                *a2 = instr->operand[2];
    
    cphvb_intp  j, off0, off1, off2,            // Index and stride offset pointers
                start0, start1, start2,         // View offset in elements.
                *stride0, *stride1, *stride2;   // Pointers to operand strides

    cphvb_index coord[CPHVB_MAXDIM],            // Coordinate map, for traversing arrays
                nelements = cphvb_nelements( a0->ndim, a0->shape ), // elements
                ec = 0,                         // elements counted
                last_dim = a0->ndim-1;          // 

    memset(coord, 0, CPHVB_MAXDIM * sizeof(cphvb_index));

                                            // Assuming that the first operand is an array.
    if(cphvb_malloc_array_data(a0) != CPHVB_SUCCESS) {
        instr->status = CPHVB_OUT_OF_MEMORY;
        return CPHVB_PARTIAL_SUCCESS;
    }

    d0      = (T*)cphvb_base_array(instr->operand[0])->data;
    stride0 = a0->stride;
    start0  = a0->start;

    if(a1 == CPHVB_CONSTANT) {
        d1 = (T*) &instr->constant[1];
        stride1 = const_stride;
        start1  = 0;
    } else {
        if(cphvb_malloc_array_data(a1) != CPHVB_SUCCESS) {
            instr->status = CPHVB_OUT_OF_MEMORY;
            return CPHVB_PARTIAL_SUCCESS;
        }

        d1 = (T*) cphvb_base_array(instr->operand[1])->data;
        stride1 = a1->stride;
        start1  = a1->start;
    }

    if(a2 == CPHVB_CONSTANT) {
        d2 = (T*) &instr->constant[2];
        stride2 = const_stride;
        start2  = 0;
    } else {
        if(cphvb_malloc_array_data(a2) != CPHVB_SUCCESS) {
            instr->status = CPHVB_OUT_OF_MEMORY;
            return CPHVB_PARTIAL_SUCCESS;
        }

        d2 = (T*) cphvb_base_array(instr->operand[2])->data;
        stride2 = a2->stride;
        start2  = a2->start;
    }

    while( ec < nelements ) {

        for(    off0 = start0,                  // Calculate offset based on coordinates
                off1 = start1,                  // INIT
                off2 = start2,                  //
                j=0;                            //

            j<last_dim;                         // COND

            ++j) {                              // INCR

            off0 += coord[j] * stride0[j];      // BODY
            off1 += coord[j] * stride1[j];
            off2 += coord[j] * stride2[j];

        }
       
        for(    coord[last_dim]=0;                      // Loop over last dimension
                coord[last_dim] < a0->shape[last_dim];

                coord[last_dim]++,
                off0 += stride0[last_dim],
                off1 += stride1[last_dim],
                off2 += stride2[last_dim]

                ) {
                                                    // Call element-wise operation

            opcode_func( (off0+d0), (off1+d1), (off2+d2) );

        }

        ec += a0->shape[last_dim];

        for(j=a0->ndim-2; j >= 0; j--) {
            coord[j]++;
            if (coord[j] < a0->shape[j]) {
                break;
            } else {
                coord[j] = 0;
            }
        }

    }
    
    return CPHVB_SUCCESS;

}

template <typename T, typename Instr>
cphvb_error traverse_2( cphvb_instruction *instr ) {

    Instr opcode_func;

    T *d0, *d1;                            // Pointers to start of data elements
    cphvb_array *a0 = instr->operand[0],        // Operands
                *a1 = instr->operand[1];
    
    cphvb_intp  j, off0, off1,              // Index and stride offset pointers
                start0, start1,             // View offset in elements.
                *stride0, *stride1;         // Pointers to operand strides

    cphvb_index coord[CPHVB_MAXDIM],            // Coordinate map, for traversing arrays
                nelements = cphvb_nelements( a0->ndim, a0->shape ), // elements
                ec = 0,                         // elements counted
                last_dim = a0->ndim-1;          // 

    memset(coord, 0, CPHVB_MAXDIM * sizeof(cphvb_index));

                                            // Assuming that the first operand is an array.
    if(cphvb_malloc_array_data(a0) != CPHVB_SUCCESS) {
        instr->status = CPHVB_OUT_OF_MEMORY;
        return CPHVB_PARTIAL_SUCCESS;
    }

    d0      = (T*)cphvb_base_array(instr->operand[0])->data;
    stride0 = a0->stride;
    start0  = a0->start;

    if(a1 == CPHVB_CONSTANT) {
        d1 = (T*) &instr->constant[1];
        stride1 = const_stride;
        start1  = 0;
    } else {
        if(cphvb_malloc_array_data(a1) != CPHVB_SUCCESS) {
            instr->status = CPHVB_OUT_OF_MEMORY;
            return CPHVB_PARTIAL_SUCCESS;
        }

        d1 = (T*) cphvb_base_array(instr->operand[1])->data;
        stride1 = a1->stride;
        start1  = a1->start;
    }

    while( ec < nelements ) {

        for(    off0 = start0,                  // Calculate offset based on coordinates
                off1 = start1,                  // INIT
                j=0;                            //

            j<last_dim;                         // COND

            ++j) {                              // INCR

            off0 += coord[j] * stride0[j];      // BODY
            off1 += coord[j] * stride1[j];

        }

        for(    coord[last_dim]=0;                      // Loop over last dimension
                coord[last_dim] < a0->shape[last_dim];

                coord[last_dim]++,
                off0 += stride0[last_dim],
                off1 += stride1[last_dim]

                ) {
                                                    // Call element-wise operation
            opcode_func( (off0+d0), (off1+d1) );

        }
        ec += a0->shape[last_dim];

        for(j=a0->ndim-2; j >= 0; j--) {
            coord[j]++;
            if (coord[j] < a0->shape[j]) {
                break;
            } else {
                coord[j] = 0;
            }
        }

    }
    
    return CPHVB_SUCCESS;

}

template <typename T, typename Instr>
cphvb_error traverse_1( cphvb_instruction *instr ) {

    Instr opcode_func;

    T *d0;                            // Pointers to start of data elements
    cphvb_array *a0 = instr->operand[0];        // Operands
    
    cphvb_intp  j, off0,            // Index and stride offset pointers
                start0,          // View offset in elements.
                *stride0;   // Pointers to operand strides

    cphvb_index coord[CPHVB_MAXDIM],            // Coordinate map, for traversing arrays
                nelements = cphvb_nelements( a0->ndim, a0->shape ), // elements
                ec = 0,                         // elements counted
                last_dim = a0->ndim-1;          // 

    memset(coord, 0, CPHVB_MAXDIM * sizeof(cphvb_index));

                                            // Assuming that the first operand is an array.
    if(cphvb_malloc_array_data(a0) != CPHVB_SUCCESS) {
        instr->status = CPHVB_OUT_OF_MEMORY;
        return CPHVB_PARTIAL_SUCCESS;
    }

    d0      = (T*)cphvb_base_array(instr->operand[0])->data;
    stride0 = a0->stride;
    start0  = a0->start;

    while( ec < nelements ) {

        for(    off0 = start0,                  // Calculate offset based on coordinates
                j=0;                            //

            j<last_dim;                         // COND

            ++j) {                              // INCR

            off0 += coord[j] * stride0[j];      // BODY

        }

        for(    coord[last_dim]=0;                      // Loop over last dimension
                coord[last_dim] < a0->shape[last_dim];

                coord[last_dim]++,
                off0 += stride0[last_dim]

                ) {
                                                    // Call element-wise operation
            opcode_func( (off0+d0) );

        }
        ec += a0->shape[last_dim];

        for(j=a0->ndim-2; j >= 0; j--) {
            coord[j]++;
            if (coord[j] < a0->shape[j]) {
                break;
            } else {
                coord[j] = 0;
            }
        }

    }
    
    return CPHVB_SUCCESS;

}

