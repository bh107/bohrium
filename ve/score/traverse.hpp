#include <cstring>
#include <iostream>
#include <cphvb.h>

void pp_instr( cphvb_instruction *instr ) {

    cphvb_array * op[3];

    op[0] = instr->operand[0];
    op[1] = instr->operand[1];
    op[2] = instr->operand[2];

    for(int j=0; j<3; j++) {

        std::cout << "Op" << j << " {" << std::endl;
        std::cout << "\tDims:\t"    << op[j]->ndim << std::endl;
        std::cout << "\tStart:\t"   << op[j]->start << std::endl;
        std::cout << "\tShape:\t";
        for(int i=0; i< op[j]->ndim; i++) {
            std::cout << op[j]->shape[i];
            if (i<op[j]->ndim-1) {
                std::cout << ",";
            }
        }
        std::cout << "." << std::endl;

        std::cout << "\tStride:\t";
        for(int i=0; i< op[j]->ndim; i++) {
            std::cout << op[j]->stride[i];
            if (i<op[j]->ndim-1) {
                std::cout << ",";
            }
        }
        std::cout << "." << std::endl;

        std::cout << "};" << std::endl;

    }

}

template <typename T, typename Instr>
cphvb_error traverse_3( cphvb_instruction *instr ) {

    Instr opcode_func;

    T *d0, *d1, *d2;                            // Pointers to start of data elements
    cphvb_array *a0 = instr->operand[0],        // Operands
                *a1 = instr->operand[1],
                *a2 = instr->operand[2];

    //cphvb_intp  j, off0, off1, off2;            // Index and stride offset pointers
    cphvb_intp j, off0, off1, off2;            // Index and stride offset pointers

    cphvb_index nelements = cphvb_nelements( a0->ndim, a0->shape ), // elements
                ec = 0,                         // elements counted
                last_dim = a0->ndim-1;          //

    cphvb_index coord[CPHVB_MAXDIM];            // Coordinate map, for traversing arrays
    memset(coord, 0, CPHVB_MAXDIM * sizeof(cphvb_index));

                                                // Assuming that the first operand is an array.
    if(cphvb_malloc_array_data(a0) != CPHVB_SUCCESS) {
        instr->status = CPHVB_OUT_OF_MEMORY;
        return CPHVB_PARTIAL_SUCCESS;
    }
    d0 = (T*)cphvb_base_array(instr->operand[0])->data;

    if(cphvb_malloc_array_data(a1) != CPHVB_SUCCESS) {
        instr->status = CPHVB_OUT_OF_MEMORY;
        return CPHVB_PARTIAL_SUCCESS;
    }
    d1 = (T*) cphvb_base_array(instr->operand[1])->data;

    if(cphvb_malloc_array_data(a2) != CPHVB_SUCCESS) {
        instr->status = CPHVB_OUT_OF_MEMORY;
        return CPHVB_PARTIAL_SUCCESS;
    }
    d2 = (T*) cphvb_base_array(instr->operand[2])->data;

    //pp_instr( instr );

    while( ec < nelements ) {
        
        for( j=0,
                off0 = a0->start,               // Calculate offset based on coordinates
                off1 = a1->start,               // INIT
                off2 = a2->start                //
                ;                            //

            j<last_dim;                         // COND

            ++j) {                              // INCR

            off0 += coord[j] * a0->stride[j];   // BODY
            off1 += coord[j] * a1->stride[j];
            off2 += coord[j] * a2->stride[j];

        }

        for(    coord[last_dim]=0;              // Loop over last dimension
                coord[last_dim] < a0->shape[last_dim];

                coord[last_dim]++,
                off0 += a0->stride[last_dim],
                off1 += a1->stride[last_dim],
                off2 += a2->stride[last_dim]

                ) {
                                                    // Call element-wise operation
            opcode_func( (off0+d0), (off1+d1), (off2+d2) );

        }

        ec += a0->shape[last_dim];

        for(j = a0->ndim-2; j >= 0; j--) {
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

    T *d0, *d1;                                 // Pointers to start of data elements
    cphvb_array *a0 = instr->operand[0],        // Operands
                *a1 = instr->operand[1];
    
    cphvb_intp  j, off0, off1;                  // Index and stride offset pointers

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
    d0 = (T*)cphvb_base_array(instr->operand[0])->data;

    if(cphvb_malloc_array_data(a1) != CPHVB_SUCCESS) {
        instr->status = CPHVB_OUT_OF_MEMORY;
        return CPHVB_PARTIAL_SUCCESS;
    }
    d1 = (T*) cphvb_base_array(instr->operand[1])->data;

    while( ec < nelements ) {

        for(    off0 = a0->start,               // Calculate offset based on coordinates
                off1 = a1->start,               // INIT
                j=0;                            //

            j<last_dim;                         // COND

            ++j) {                              // INCR

            off0 += coord[j] * a0->stride[j];   // BODY
            off1 += coord[j] * a1->stride[j];

        }

        for(    coord[last_dim]=0;                  // Loop over last dimension
                coord[last_dim] < a0->shape[last_dim];

                coord[last_dim]++,
                off0 += a0->stride[last_dim],
                off1 += a1->stride[last_dim]

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

    T *d0;                                      // Pointers to start of data elements
    cphvb_array *a0 = instr->operand[0];        // Operands
    
    cphvb_intp  j, off0;                        // Index and stride offset pointers

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

    while( ec < nelements ) {

        for(    off0 = a0->start,               // Calculate offset based on coordinates
                j=0;                            //

            j<last_dim;                         // COND

            ++j) {                              // INCR

            off0 += coord[j] * a0->stride[j];   // BODY

        }

        for(    coord[last_dim]=0;              // Loop over last dimension
                coord[last_dim] < a0->shape[last_dim];

                coord[last_dim]++,
                off0 += a0->stride[last_dim]

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

