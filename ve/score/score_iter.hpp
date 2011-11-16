#include <cstring>
#include <iostream>
#include <cphvb.h>

void pp_array_data( ) {

}

void pp_array_meta( cphvb_array *array ) {

    if (array != CPHVB_CONSTANT) {

        std::cout << "cphvb_array ID: " << array << " {" << std::endl;
        std::cout << "\towner: "        << array->owner << std::endl;
        std::cout << "\tbase: "         << array->base << std::endl;
        if (array->base != NULL) {
            std::cout << "\tbase->data: "<< array->base->data << std::endl;
        }
        std::cout << "\ttype: "         << cphvb_type_text(array->type) << std::endl;
        std::cout << "\tndim: "         << array->ndim << std::endl;
        std::cout << "\tstart: "        << array->start << std::endl;

        for (int i = 0; i < array->ndim; ++i) {
            std::cout << "\tshape["<<i<<"]: " << array->shape[i] << std::endl;
        }
        for (int i = 0; i < array->ndim; ++i) {
            std::cout << "\tstride["<<i<<"]: " << array->stride[i] << std::endl;
        }
        std::cout << "\tdata: " << array->data << std::endl;
        std::cout << "\thas_init_value: " << array->has_init_value << std::endl;
        switch(array->type) {
            case CPHVB_INT32:
                std::cout << "\tinit_value: " << array->init_value.int32 << std::endl;
                break;
            case CPHVB_UINT32:
                std::cout << "\tinit_value: " << array->init_value.uint32 << std::endl;
                break;
            case CPHVB_FLOAT32:
                std::cout << "\tinit_value: " << array->init_value.float32 << std::endl;
                break;
        }
        std::cout << "}"<< std::endl;

    } else {

        std::cout << "operand is a CONSTANT not a cphvb_array. { ... }" << std::endl;

    }

}

void pp_instr_meta( cphvb_instruction *instr ) {

    std::cout << "cphvb_instruction ID: " << instr << " {" << std::endl;
    std::cout << "\tstatus: " << instr->status << std::endl;
    std::cout << "\topcode: " << instr->opcode << std::endl;

    for(int i=0; i<cphvb_operands( instr->opcode ); i++) {
        pp_array_meta( instr->operand[i] );
    }
    
    std::cout << "}" << std::endl;

}

template <typename T>
inline cphvb_error iter(cphvb_instruction *instr, cphvb_error (*opcode_func)(T*, T*, T*)) {

    
    T           *d0,    *d1,    *d2;        // Pointers to the first data-element.
    cphvb_array *a0,    *a1,    *a2;        // Pointers to array/view meta data.
    cphvb_intp  start0, start1, start2;     // View-offset in elements.
    cphvb_intp  off0,   off1,   off2;       // Offset from first data-element to reach the current
    cphvb_intp  step0,  step1,  step2;      // Elements to skip to reach next data-element
    cphvb_intp  *stride0, *stride1, *stride2;

    cphvb_intp j, finished=0;
    cphvb_index coord[CPHVB_MAXDIM];
    memset(coord, 0, CPHVB_MAXDIM * sizeof(cphvb_index));

    /*
    cphvb_intp ct, cd;                      // Count total number of elements
                                            // Count elements in current dimension
*/    


    cphvb_intp const_stride[CPHVB_MAXDIM] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

    a0 = instr->operand[0];                 //
    a1 = instr->operand[1];                 // Determine and Verify operands
    a2 = instr->operand[2];                 //

    if (a0 == CPHVB_CONSTANT) {             // Constant operand

        d0      = (T*)&instr->constant[0];
        step0   = 0;
        start0  = 0;
        stride0 = const_stride;

    } else {                                // Array operand

        if ((a0->base == NULL) && (a0->data == NULL)) {             // Unallocated array

            if(cphvb_malloc_array_data(a0) != CPHVB_SUCCESS) {
                fprintf(stderr,"Out of memory attempting to perform instruction.\n");
                return CPHVB_OUT_OF_MEMORY;
            }
        }

        d0      = (T*)cphvb_base_array(a0)->data;
        step0   = 1;
        start0  = a0->start;
        stride0 = a0->stride;

    }

    if (a1 == CPHVB_CONSTANT) {             // Constant operand

        d1      = (T*)&instr->constant[1];
        step1   = 0;
        start1  = 0;
        stride1 = const_stride;

    } else {                                // Array operand

        if ((a1->base == NULL) && (a1->data == NULL)) {             // Unallocated array

            if(cphvb_malloc_array_data(a1) != CPHVB_SUCCESS) {
                fprintf(stderr,"Out of memory attempting to perform instruction.\n");
                return CPHVB_OUT_OF_MEMORY;
            }
        }

        d1      = (T*)cphvb_base_array(a1)->data;
        step1   = 1;
        start1  = a1->start;
        stride1 = a1->stride;

    }

    if (a2 == CPHVB_CONSTANT) {             // Constant operand

        d2      = (T*)&instr->constant[2];
        step2   = 0;
        start2  = 0;
        stride2 = const_stride;

    } else {                                // Array operand

        if ((a2->base == NULL) && (a2->data == NULL)) {             // Unallocated array

            if(cphvb_malloc_array_data(a2) != CPHVB_SUCCESS) {
                fprintf(stderr,"Out of memory attempting to perform instruction.\n");
                return CPHVB_OUT_OF_MEMORY;
            }
        }

        d2      = (T*)cphvb_base_array(a2)->data;
        step2   = 1;
        start2  = a2->start;
        stride2 = a2->stride;

    }

    while(!finished) {

        off0 = start0;
        for(j=0; j<a0->ndim; ++j) {
            off0 += coord[j] * stride0[j];
        }

        off1 = start1;
        for(j=0; j<a0->ndim; ++j) {
            off1 += coord[j] * stride1[j];
        }

        off2 = start2;
        for(j=0; j<a0->ndim; ++j) {
            off2 += coord[j] * stride2[j];
        }

        (*opcode_func)( (off0+d0), (off1+d1), (off2+d2) );

        for(j=a0->ndim-1; j >=0; j--) {
            if(++coord[j] >= a0->shape[j]) {
                if(j==0) {
                    finished = true;
                    break;
                }
                coord[j] = 0;
            } else {
                break;
            }
        }

    }

    //
    // array, array => array
    // array, const => array
    // const, array => array
    //

    /*
    pp_instr_meta( instr );

    cphvb_intp undim = 0;

    off0 = start0;
    off1 = start1;
    off2 = start2;

    for( cphvb_intp dim=0; dim < a0->ndim; dim++ ) {


        undim = a0->ndim-dim;

        for(                            // Init
            cd=0,
            step0 = stride0[dim],
            step1 = stride1[dim],
            step2 = stride2[dim];

            cd < a0->shape[dim];        // Condition

            cd++,
            ct++,
            off0 += step0,              // Increment
            off1 += step1,
            off2 += step2
            ) {

            std::cout << dim << ": " << off0 << "," << off1 << "," << off2 << "\n" << std::endl;
            (*opcode_func)( off0+d0, off1+d1, off2+d2 );
        }

    }
    */
    /*
    for( cphvb_intp dim=0; dim < a0->ndim; dim++ ) {

        for(                                // Init
            step0 = stride0[dim],
            step1 = stride1[dim],
            step2 = stride2[dim],
            elements += a0->shape[dim]*a0->stride[dim];

            off0 < elements;                // Condition

            off0 += step0,                  // Increment
            off1 += step1,
            off2 += step2 ) {

            std::cout << dim << "," << off0 << "," << off1 << "," << off2 << "\n" << std::endl;
            (*opcode_func)( off0+d0, off1+d1, off2+d2 );

        }
    }
    */

    return CPHVB_SUCCESS;

}

template <typename T>
cphvb_error iter(cphvb_instruction *instr, cphvb_error (*opcode_func)(T*, T*)) {
    
    cphvb_intp j, notfinished=1;

    T *d0, *d1;
    cphvb_array *a0 = instr->operand[0];
    cphvb_array *a1 = instr->operand[1];
    cphvb_index coord[CPHVB_MAXDIM];
    std::memset(coord, 0, CPHVB_MAXDIM * sizeof(cphvb_index));

    cphvb_error res = CPHVB_SUCCESS;

    //
    // Determine and Verify operands
    //
    if (a0 == CPHVB_CONSTANT) {             // Constant operand
        d0 = (T*)&instr->constant[1];

    } else if (a0->data == NULL) {          // Unallocated array-operand

        if(cphvb_malloc_array_data(a0) != CPHVB_SUCCESS) {
            fprintf(stderr,"Out of memory attempting to perform instruction.\n");
            return CPHVB_OUT_OF_MEMORY;
        }
        d0 = (T*)cphvb_base_array(a0)->data;

    } else {                                // Allocated array-operand
        d0 = (T*)cphvb_base_array(a0)->data;
    }

    if (a1 == CPHVB_CONSTANT) {             // Constant operand
        d1 = (T*)&instr->constant[1];

    } else if (a1->data == NULL) {          // Unallocated array-operand

        if(cphvb_malloc_array_data(a1) != CPHVB_SUCCESS) {
            fprintf(stderr,"Out of memory attempting to perform instruction.\n");
            return CPHVB_OUT_OF_MEMORY;
        }
        d1 = (T*)cphvb_base_array(a1)->data;

    } else {                                // Allocated array-operand
        d1 = (T*)cphvb_base_array(a1)->data;
    }

    while(notfinished) {

        cphvb_intp off0=0, off1=0;

        for(j=0; j< a0->ndim; ++j) {
            off0 += coord[j] * a0->stride[j];
        }
        off0 += a0->start;

        if(a1 != CPHVB_CONSTANT) {
            for(j=0; j<a0->ndim; ++j)
                off1 += coord[j] * a1->stride[j];
            off1 += a1->start;
        }

        res = (*opcode_func)( (d0+off0), (d1+off1) );   // Compute element

        for(j=a0->ndim-1; j >= 0; j--) {                // Iterate coord one element.

            if(++coord[j] >= a0->shape[j]) {
                
                if(j == 0) {                            // We are finished, if wrapping around.
                    notfinished = 0;
                    break;
                }
                coord[j] = 0;
            } else {
                break;
            }

        }
    }

    return res;

}

template <typename T>
cphvb_error iter(cphvb_instruction *instr, cphvb_error (*opcode_func)(T*)) {
    
    cphvb_intp j, notfinished=1;
    T *d0;
    cphvb_array *a0 = instr->operand[0];
    cphvb_index coord[CPHVB_MAXDIM];
    std::memset(coord, 0, CPHVB_MAXDIM * sizeof(cphvb_index));
    
    cphvb_error res = CPHVB_SUCCESS;

    //
    // Determine and Verify operands
    //
    if (a0 == CPHVB_CONSTANT) {             // Constant operand
        d0 = (T*)&instr->constant[1];

    } else if (a0->data == NULL) {          // Unallocated array-operand

        if(cphvb_malloc_array_data(a0) != CPHVB_SUCCESS) {
            fprintf(stderr,"Out of memory attempting to perform instruction.\n");
            return CPHVB_OUT_OF_MEMORY;
        }
        d0 = (T*)cphvb_base_array(a0)->data;

    } else {                                // Allocated array-operand
        d0 = (T*)cphvb_base_array(a0)->data;
    }

    while(notfinished) {

        cphvb_intp off0=0;

        for(j=0; j< a0->ndim; ++j) {
            off0 += coord[j] * a0->stride[j];
        }
        off0 += a0->start;

        res = (*opcode_func)( (d0+off0) );      // Compute element

        for(j=a0->ndim-1; j >= 0; j--) {        // Iterate coord one element.

            if(++coord[j] >= a0->shape[j]) {
                
                if(j == 0) {                    // We are finished, if wrapping around.
                    notfinished = 0;
                    break;
                }
                coord[j] = 0;
            } else {
                break;
            }

        }
    }

    return res;

}

