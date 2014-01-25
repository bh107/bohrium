{
    /*
    int reduction(
        int tool,

        T       *a0_first,
        int64_t  a0_start,
        int64_t *a0_stride,
        int64_t *a1_shape,
        int64_t  a1_ndim,

        T       *a1_first,
        int64_t  a1_start,
        int64_t *a1_stride,
        int64_t *a1_shape,
        int64_t  a1_ndim,

        T       *a2_first
    )
    */

    va_list list;                                   // **UNPACK PARAMETERS**
    va_start(list, tool);

    {{#OPERAND}}
    {{TYPE}} *a{{NR}}_first   = va_arg(list, {{TYPE}}*);
    {{#ARRAY}}
    int64_t  a{{NR}}_start   = va_arg(list, int64_t);
    int64_t *a{{NR}}_stride  = va_arg(list, int64_t*);
    int64_t *a{{NR}}_shape   = va_arg(list, int64_t*);
    int64_t  a{{NR}}_ndim    = va_arg(list, int64_t);
    {{TYPE}} *a{{NR}}_current = a{{NR}}_first + a{{NR}}_start;
    {{/ARRAY}}
    {{/OPERAND}}

    va_end(list);                                   // **DONE UNPACKING**

    {{TYPE_AXIS}} axis = *a2_first;                 // Use the first element as temp

    int64_t a1_i;               // Iterator variables...

    {{TYPE_INPUT}} *tmp_current;    // Intermediate array
    {{TYPE_INPUT}} *tmp_first;      
    int64_t tmp_start;
    int64_t tmp_stride[CPU_MAXDIM];    

    if (1 == a1_ndim) {                             // ** 1D Special Case **
        a0_current = a0_first + a0_start;           // Point to first element in output.
        {{TYPE_INPUT}} rvar = *(a1_first+a1_start);    // Use the first element as temp
        for(tmp_current = a1_first+a1_start+a1_stride[axis], a1_i=1;
            a1_i < a1_shape[axis];
            tmp_current += a1_stride[axis], a1_i++) {
            
            {{OPERATOR}};
        }
        *a0_current = rvar;
    } else {                                    // ** ND General Case **
        int64_t j,                              // Traversal variables
                last_dim,
                last_e,
                cur_e,
                coord[CPU_MAXDIM];

        tmp_first   = a1_first;                  // Use the temporary as a copy of input
        tmp_start   = a1_start;                 // without the 'axis' dimension

        int64_t tmp_dim;
        for (tmp_dim=0, a1_i=0; a1_i<a1_ndim; ++a1_i) { // Excluding the 'axis' dimension.
            if (a1_i != axis) {
                tmp_stride[tmp_dim]   = a1_stride[a1_i];
                ++tmp_dim;
            }
        }

        last_e = 1;
        int64_t k;
        for (k = 0; k < a0_ndim; ++k) { // COUNT THE ELEMENTS
            last_e *= a0_shape[k];
        }
        --last_e;

        last_dim = a0_ndim-1;

        for(a1_i=0; a1_i<a1_shape[axis]; ++a1_i, tmp_start += a1_stride[axis]) {

            cur_e = 0;                                  // Reset coordinate and element counter
            memset(coord, 0, CPU_MAXDIM * sizeof(int64_t));

            while (cur_e <= last_e) {
                a0_current   = a0_first + a0_start;       // Reset offsets
                tmp_current  = tmp_first + tmp_start;

                for (j=0; j<=last_dim; ++j) {           // Compute offset based on coordinate
                    a0_current   += coord[j] * a0_stride[j];
                    tmp_current  += coord[j] * tmp_stride[j];
                }
                                                        // Iterate over "last" / "innermost" dimension
                if (0==a1_i) {                         // First off, copy the intermediate value
                    for(;
                        (coord[last_dim] < a0_shape[last_dim]) && (cur_e <= last_e);
                        coord[last_dim]++,                  // Increment coordinates
                        cur_e++
                    ) {
                        *a0_current = *tmp_current;

                        a0_current   += a0_stride[last_dim]; // Increment element indexes
                        tmp_current  += tmp_stride[last_dim];
                    }
                } else {                                // Then do the actual reduction
                    for(;
                        (coord[last_dim] < a0_shape[last_dim]) && (cur_e <= last_e);
                        coord[last_dim]++,              // Coordinates
                        cur_e++
                    ) {
                        {{TYPE_INPUT}} rvar = *a0_current; // Scalar-temp
                        {{OPERATOR}};
                        *a0_current = rvar;

                        a0_current   += a0_stride[last_dim]; // Offsets
                        tmp_current  += tmp_stride[last_dim];
                    }
                }

                if (coord[last_dim] >= a0_shape[last_dim]) {
                    coord[last_dim] = 0;
                    for(j = last_dim-1; j >= 0; --j) {  // Increment coordinates for the remaining dimensions
                        coord[j]++;
                        if (coord[j] < a0_shape[j]) {   // Still within this dimension
                            break;
                        } else {                        // Reached the end of this dimension
                            coord[j] = 0;               // Reset coordinate
                        }                               // Loop then continues to increment the next dimension
                    }
                }
            }
        }
    }
}

