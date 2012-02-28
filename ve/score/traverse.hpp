#include <cstring>
#include <iostream>
#include <cphvb.h>

template <typename T0, typename T1, typename T2, typename Instr>
cphvb_error traverse_3( cphvb_instruction *instr ) {

    T0 *d0; T1 *d1; T2 *d2;                     // Pointers to start of data elements
    cphvb_array *a0 = instr->operand[0],        // Operands
                *a1 = instr->operand[1],
                *a2 = instr->operand[2];
    Instr opcode_func;

    //TODO:  handle scalars
    d0 = (T0*) cphvb_base_array(instr->operand[0])->data;
    d1 = (T1*) cphvb_base_array(instr->operand[1])->data;
    d2 = (T2*) cphvb_base_array(instr->operand[2])->data;

    cphvb_index last_dim = a0->ndim-1;
    cphvb_intp j, off0, off1, off2;             // Index and stride offset pointers
    cphvb_index coord[CPHVB_MAXDIM];
    memset(coord, 0, CPHVB_MAXDIM * sizeof(cphvb_index));

    int notfinished = 1;
    while( notfinished ) {
        off0 = a0->start + coord[0] * a0->stride[0];
        off1 = a1->start + coord[0] * a1->stride[0];
        off2 = a2->start + coord[0] * a2->stride[0];
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
            if(j==0 && coord[j] >= a0->shape[0])
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

    return CPHVB_SUCCESS;

}

template <typename T0,typename T1, typename Instr>
cphvb_error traverse_2( cphvb_instruction *instr ) {

    Instr opcode_func;

    T0 *d0; T1 *d1;                             // Pointers to start of data elements
    cphvb_array *a0 = instr->operand[0],        // Operands
                *a1 = instr->operand[1];

    cphvb_intp  j, off0, off1;                  // Index and stride offset pointers

    cphvb_index coord[CPHVB_MAXDIM],            // Coordinate map, for traversing arrays
                nelements = cphvb_nelements( a0->ndim, a0->shape ), // elements
                ec = 0,                         // elements counted
                last_dim = a0->ndim-1;          //

    memset(coord, 0, CPHVB_MAXDIM * sizeof(cphvb_index));

    //TODO:  handle scalars
    d0 = (T0*) cphvb_base_array(instr->operand[0])->data;
    d1 = (T1*) cphvb_base_array(instr->operand[1])->data;

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

