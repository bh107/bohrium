#include <cstring>
#include <iostream>
#include <cphvb.h>

#ifdef _OPENMP
#include <omp.h>
#endif

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

template <typename Tout, typename Tin, typename Instr>
cphvb_error traverse_3( cphvb_instruction *instr ) {

    Tin *d0;                                    // Pointers to start of data elements
    Tout *d1, *d2;
    cphvb_array *a0 = instr->operand[0],        // Operands
                *a1 = instr->operand[1],
                *a2 = instr->operand[2];

    cphvb_intp nthds = 1;
                                                // Assuming that the first operand is an array.
    if(cphvb_malloc_array_data(a0) != CPHVB_SUCCESS) {
        instr->status = CPHVB_OUT_OF_MEMORY;
        return CPHVB_PARTIAL_SUCCESS;
    }
    d0 = (Tin*)cphvb_base_array(instr->operand[0])->data;

    if(cphvb_malloc_array_data(a1) != CPHVB_SUCCESS) {
        instr->status = CPHVB_OUT_OF_MEMORY;
        return CPHVB_PARTIAL_SUCCESS;
    }
    d1 = (Tout*) cphvb_base_array(instr->operand[1])->data;

    if(cphvb_malloc_array_data(a2) != CPHVB_SUCCESS) {
        instr->status = CPHVB_OUT_OF_MEMORY;
        return CPHVB_PARTIAL_SUCCESS;
    }
    d2 = (Tout*) cphvb_base_array(instr->operand[2])->data;

    //We will use OpenMP to parallelize of the computation.
    //We divide the work over the first dimension, i.e. the most
    //significant dimension.
    #ifdef _OPENMP
        if(a0->ndim > 1) //Find number of threads to use.
        {
            nthds = omp_get_max_threads();
            if(nthds > a0->shape[0])
                nthds = a0->shape[0];//Minimum one element per thread.
        }
        #pragma omp parallel num_threads(nthds) default(none) shared(nthds,a0,a1,a2,d0,d1,d2)
    #endif
    {
        Instr opcode_func;
        #ifdef _OPENMP
            int myid = omp_get_thread_num();
        #else
            int myid = 0;
        #endif
        cphvb_index last_dim = a0->ndim-1;
        cphvb_intp j, off0, off1, off2;             // Index and stride offset pointers
        cphvb_index coord[CPHVB_MAXDIM];
        memset(coord, 0, CPHVB_MAXDIM * sizeof(cphvb_index));
        cphvb_intp length = a0->shape[0] / nthds;   // Find this thread's length of work.
        cphvb_intp thd_offset = myid * length;      // Find this thread's offset.
        if(myid == nthds-1)
            length += a0->shape[0] % nthds;         // The last thread get the rest.

        int notfinished = 1;
        while( notfinished ) {
            off0 = thd_offset * a0->stride[0] + a0->start + coord[0] * a0->stride[0];
            off1 = thd_offset * a1->stride[0] + a1->start + coord[0] * a1->stride[0];
            off2 = thd_offset * a2->stride[0] + a2->start + coord[0] * a2->stride[0];
            for( j=1; j<last_dim; ++j) {

                off0 += coord[j] * a0->stride[j];
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

            for(j = last_dim; j >= 0; --j) {
                coord[j]++;
                if(j==0 && coord[j] >= length)
                {
                    notfinished = 0;
                    break;
                }
                else if (coord[j] < a0->shape[j]) {
                    break;
                } else {
                    coord[j] = 0;
                }
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

