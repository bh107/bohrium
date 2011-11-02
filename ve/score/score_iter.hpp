#include <cstring>
#include <iostream>
#include <cphvb.h>

template <typename T>
cphvb_error iter(cphvb_instruction *instr, cphvb_error (*opcode_func)(T*, T*, T*)) {
    
    cphvb_intp j, notfinished=1;
    T *d0, *d1, *d2;
    cphvb_array *a0 = instr->operand[0];
    cphvb_array *a1 = instr->operand[1];
    cphvb_array *a2 = instr->operand[2];
    cphvb_index coord[CPHVB_MAXDIM];
    std::memset(coord, 0, CPHVB_MAXDIM * sizeof(cphvb_index));

    std::cout << "Number of dimensions " << a0->ndim << " total noe " << cphvb_nelements(a0->ndim, a0->shape) << "\n" << std::endl;
    for(cphvb_index k=0; k<= a0->ndim; k++) {
        std::cout << "Dim " << k << " has " << a0->shape[k] << " elements.\n" << std::endl;
    }
    std::cout << ".\n" << std::endl;
    
    cphvb_error res = CPHVB_SUCCESS;
    
    if (a0->data == NULL) {
        if(cphvb_malloc_array_data(a0) != CPHVB_SUCCESS) {
            fprintf(stderr,"Out of memory attempting to perform instruction.\n");
            return CPHVB_OUT_OF_MEMORY;
        }
    }
    if (a1->data == NULL) {
        if(cphvb_malloc_array_data(a1) != CPHVB_SUCCESS) {
            fprintf(stderr,"Out of memory attempting to perform instruction.\n");
            return CPHVB_OUT_OF_MEMORY;
        }
    }
    if (a2->data == NULL) {
        if(cphvb_malloc_array_data(a2) != CPHVB_SUCCESS) {
            fprintf(stderr,"Out of memory attempting to perform instruction.\n");
            return CPHVB_OUT_OF_MEMORY;
        }
    }

    d0 = (T*)cphvb_base_array(instr->operand[0])->data;

    if (a1 == CPHVB_CONSTANT) {
        d1 = (T*)&instr->constant[1];
    } else {
        d1 = (T*)cphvb_base_array(instr->operand[1])->data;
    }

    if(a2 == CPHVB_CONSTANT) {
        d2 = (T*)&instr->constant[2];
    } else {
        d2 = (T*)cphvb_base_array(instr->operand[2])->data;
    }

    while(notfinished) {

        cphvb_intp off0=0, off1=0, off2=0;

        for(j=0; j<a0->ndim; ++j) {
            off0 += coord[j] * a0->stride[j];
        }
        off0 += a0->start;

        if(a1 != CPHVB_CONSTANT) {
            for(j=0; j<a0->ndim; ++j)
                off1 += coord[j] * a1->stride[j];
            off1 += a1->start;
        }

        if(a2 != CPHVB_CONSTANT) {
            for(j=0; j<a0->ndim; ++j)
                off2 += coord[j] * a2->stride[j];
            off2 += a2->start;
        }
        
        res = (*opcode_func)( (d0+off0), (d1+off1), (d2+off2) );

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

template <typename T>
cphvb_error iter(cphvb_instruction *instr, cphvb_error (*opcode_func)(T*, T*)) {
    
    cphvb_intp j, notfinished=1;

    T *d0, *d1;
    cphvb_array *a0 = instr->operand[0];
    cphvb_array *a1 = instr->operand[1];
    cphvb_index coord[CPHVB_MAXDIM];
    std::memset(coord, 0, CPHVB_MAXDIM * sizeof(cphvb_index));

    cphvb_error res = CPHVB_SUCCESS;

    if (cphvb_malloc_array_data(a0) != CPHVB_SUCCESS) {
        fprintf(stderr,"Out of memory applying something\n");
        return CPHVB_OUT_OF_MEMORY;
    }
    if (cphvb_malloc_array_data(a1) != CPHVB_SUCCESS) {
        fprintf(stderr,"Out of memory applying something\n");
        return CPHVB_OUT_OF_MEMORY;
    }
    d0 = (T*)cphvb_base_array(instr->operand[0])->data;

    if (a1 == CPHVB_CONSTANT) {
        d1 = (T*) &instr->constant[1];
    } else {
        d1 = (T*) cphvb_base_array(instr->operand[1])->data;
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

    if (cphvb_malloc_array_data(a0) != CPHVB_SUCCESS) {
        fprintf(stderr,"Out of memory applying something\n");
        return CPHVB_OUT_OF_MEMORY;
    }
    d0 = (T*)cphvb_base_array(instr->operand[0])->data;

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

