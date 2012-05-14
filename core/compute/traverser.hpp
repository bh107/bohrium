#include <cphvb.h>
#include <assert.h>

template <typename T0, typename T1, typename T2, typename Instr>
cphvb_error traverse_aaa( cphvb_instruction *instr, cphvb_index skip, cphvb_index limit ) {

    Instr opcode_func;                          // Element-wise functor-pointer
    cphvb_array *a0 = instr->operand[0],        // Operands
                *a1 = instr->operand[1],
                *a2 = instr->operand[2];
                                                // Pointers to start of data elements
    T0* d0 = (T0*) cphvb_base_array(instr->operand[0])->data;
    T1* d1 = (T1*) cphvb_base_array(instr->operand[1])->data;
    T2* d2 = (T2*) cphvb_base_array(instr->operand[2])->data;

    assert(d0 != NULL);                         // Ensure that data is allocated
    assert(d1 != NULL);
    assert(d2 != NULL);

    cphvb_index j,                              // Traversal variables
                last_dim = a0->ndim-1,
                off0=skip, off1=skip, off2=skip,
                nelements = cphvb_nelements( a0->ndim, a0->shape ),
                ec = 0;

    cphvb_index coord[CPHVB_MAXDIM];
    memset(coord, 0, CPHVB_MAXDIM * sizeof(cphvb_index));

    while( ec < nelements )
    {
        off0 = a0->start + coord[0] * a0->stride[0];
        off1 = a1->start + coord[0] * a1->stride[0];
        off2 = a2->start + coord[0] * a2->stride[0];
        for( j=1; j<last_dim; ++j)
        {
            off0 += coord[j] * a0->stride[j];
            off1 += coord[j] * a1->stride[j];
            off2 += coord[j] * a2->stride[j];
        }

        for( coord[last_dim]=0; coord[last_dim] < a0->shape[last_dim]; coord[last_dim]++)
        {
            opcode_func( (off0+d0), (off1+d1), (off2+d2) );

            off0 += a0->stride[last_dim];
            off1 += a1->stride[last_dim];
            off2 += a2->stride[last_dim];
        }
        ec += a0->shape[last_dim];

        for(j = last_dim; j >= 0; --j)
        {
            coord[j]++;
            if(j==0 && coord[0] >= a0->shape[0])
            {
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

template <typename T0, typename T1, typename T2, typename Instr>
cphvb_error traverse_aac( cphvb_instruction *instr, cphvb_index skip, cphvb_index limit ) {

    Instr opcode_func;                          // Element-wise functor-pointer
    cphvb_array *a0 = instr->operand[0],        // Array-Operands
                *a1 = instr->operand[1];
                                                // Pointers to start of data elements
    T0* d0 = (T0*) cphvb_base_array(instr->operand[0])->data;
    T1* d1 = (T1*) cphvb_base_array(instr->operand[1])->data;
    T2* d2 = (T2*) &(instr->constant.value);

    assert(d0 != NULL);                         // Ensure that data is allocated
    assert(d1 != NULL);

    cphvb_index j,                              // Traversal variables
                last_dim = a0->ndim-1,
                off0=skip, off1=skip,
                nelements = cphvb_nelements( a0->ndim, a0->shape ),
                ec = 0;

    cphvb_index coord[CPHVB_MAXDIM];
    memset(coord, 0, CPHVB_MAXDIM * sizeof(cphvb_index));

    while( ec < nelements )
    {
        off0 = a0->start + coord[0] * a0->stride[0];
        off1 = a1->start + coord[0] * a1->stride[0];
        for( j=1; j<last_dim; ++j)
        {
            off0 += coord[j] * a0->stride[j];
            off1 += coord[j] * a1->stride[j];
        }

        for( coord[last_dim]=0; coord[last_dim] < a0->shape[last_dim]; coord[last_dim]++ )
        {
            opcode_func( (off0+d0), (off1+d1), d2 );

            off0 += a0->stride[last_dim];
            off1 += a1->stride[last_dim];
        }
        ec += a0->shape[last_dim];

        for(j = last_dim; j >= 0; --j)
        {
            coord[j]++;
            if(j==0 && coord[0] >= a0->shape[0])
            {
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

template <typename T0, typename T1, typename T2, typename Instr>
cphvb_error traverse_aca( cphvb_instruction *instr, cphvb_index skip, cphvb_index limit ) {

    Instr opcode_func;
    cphvb_array *a0 = instr->operand[0],        // Array-Operands
                *a2 = instr->operand[2];

    T0* d0 = (T0*) cphvb_base_array(instr->operand[0])->data;
    T1* d1 = (T1*) &(instr->constant.value);
    T2* d2 = (T2*) cphvb_base_array(instr->operand[2])->data;

    assert(d0 != NULL);
    assert(d2 != NULL);

    cphvb_index j,                              // Traversal variables
                last_dim = a0->ndim-1,
                off0=skip, off2=skip,
                nelements = cphvb_nelements( a0->ndim, a0->shape ),
                ec = 0;

    cphvb_index coord[CPHVB_MAXDIM];
    memset(coord, 0, CPHVB_MAXDIM * sizeof(cphvb_index));

    while( ec < nelements )
    {
        off0 = a0->start + coord[0] * a0->stride[0];
        off2 = a2->start + coord[0] * a2->stride[0];
        for( j=1; j<last_dim; ++j)
        {
            off0 += coord[j] * a0->stride[j];
            off2 += coord[j] * a2->stride[j];
        }

        for( coord[last_dim]=0; coord[last_dim] < a0->shape[last_dim]; coord[last_dim]++ )
        {
            opcode_func( (off0+d0), d1, (off2+d2) );

            off0 += a0->stride[last_dim];
            off2 += a2->stride[last_dim];
        }
        ec += a0->shape[last_dim];

        for(j = last_dim; j >= 0; --j)
        {
            coord[j]++;
            if(j==0 && coord[0] >= a0->shape[0])
            {
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
cphvb_error traverse_aa( cphvb_instruction *instr, cphvb_index skip, cphvb_index limit ) {

    Instr opcode_func;
    cphvb_array *a0 = instr->operand[0],        // Operands
                *a1 = instr->operand[1];

    T0* d0 = (T0*) cphvb_base_array(instr->operand[0])->data;
    T1* d1 = (T1*) cphvb_base_array(instr->operand[1])->data;

    assert(d0 != NULL);
    assert(d1 != NULL);

    cphvb_index j,                              // Traversal variables
                last_dim = a0->ndim-1,
                off0=skip, off1=skip,
                nelements = cphvb_nelements( a0->ndim, a0->shape ),
                ec = 0;

    cphvb_index coord[CPHVB_MAXDIM];
    memset(coord, 0, CPHVB_MAXDIM * sizeof(cphvb_index));
    
    while( ec < nelements )
    {
        off0 = a0->start;
        off1 = a1->start;
        for( j=0; j<last_dim; ++j )
        {
            off0 += coord[j] * a0->stride[j];
            off1 += coord[j] * a1->stride[j];
        }

        for( coord[last_dim]=0; coord[last_dim] < a0->shape[last_dim]; coord[last_dim]++ )
        {
            opcode_func( (off0+d0), (off1+d1) );

            off0 += a0->stride[last_dim];
            off1 += a1->stride[last_dim];
        }
        ec += a0->shape[last_dim];

        for(j=a0->ndim-2; j >= 0; j--)
        {
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

template <typename T0, typename T1, typename Instr>
cphvb_error traverse_ac( cphvb_instruction *instr, cphvb_index skip, cphvb_index limit ) {

    Instr opcode_func;
    cphvb_array *a0 = instr->operand[0];        // Array-Operands

    T0* d0 = (T0*) cphvb_base_array(instr->operand[0])->data;
    T1* d1 = (T1*) &(instr->constant.value);

    assert(d0 != NULL);

    cphvb_index j,                              // Traversal variables
                last_dim = a0->ndim-1,
                off0=skip,
                nelements = cphvb_nelements( a0->ndim, a0->shape ),
                ec = 0;

    cphvb_index coord[CPHVB_MAXDIM];
    memset(coord, 0, CPHVB_MAXDIM * sizeof(cphvb_index));

    while( ec < nelements )
    {
        off0 = a0->start;
        for( j=0; j<last_dim; ++j)
        {
            off0 += coord[j] * a0->stride[j];
        }

        for( coord[last_dim]=0; coord[last_dim] < a0->shape[last_dim]; coord[last_dim]++)
        {
            opcode_func( (off0+d0), d1 );

            off0 += a0->stride[last_dim];
        }
        ec += a0->shape[last_dim];

        for(j=a0->ndim-2; j >= 0; j--)
        {
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

