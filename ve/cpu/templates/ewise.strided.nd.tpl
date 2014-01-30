{
    int64_t nelements = a{{NR_OUTPUT}}_nelem;
    int mthreads = omp_get_max_threads();
    int64_t nworkers = nelements > mthreads ? mthreads : 1;

    int64_t cur_e = 0;

    {{#OPERAND}}{{#ARRAY}}
    {{TYPE}} *a{{NR}}_first   = a{{NR}}_current;
    int64_t a{{NR}}_stride_ld = a{{NR}}_stride[last_dim];
    {{/ARRAY}}{{/OPERAND}}

    int64_t coord[CPU_MAXDIM];
    memset(coord, 0, CPU_MAXDIM * sizeof(int64_t));

    while (cur_e <= last_e) {
        
        {{#OPERAND}}{{#ARRAY}}
        a{{NR}}_current = a{{NR}}_first + a{{NR}}_start;         // Reset offsets
        {{/ARRAY}}{{/OPERAND}}

        for (j=0; j<=last_dim; ++j) {           // Compute offset based on coordinate
            {{#OPERAND}}{{#ARRAY}}
            a{{NR}}_current += coord[j] * a{{NR}}_stride[j];
            {{/ARRAY}}{{/OPERAND}}
        }

        for (j = 0; j < shape_ld; j++) {        // Iterate over "last" / "innermost" dimension
            {{#LOOP_BODY}}
            {{OPERATOR}};
            {{/LOOP_BODY}}

            {{#OPERAND}}{{#ARRAY}}
            a{{NR}}_current += a{{NR}}_stride_ld;
            {{/ARRAY}}{{/OPERAND}}
        }
        cur_e += shape_ld;

        // coord[last_dim] is never used, only all the other coord[dim!=last_dim]
        for (j = last_dim-1; j >= 0; --j) {  // Increment coordinates for the remaining dimensions
            coord[j]++;
            if (coord[j] < shape[j]) {      // Still within this dimension
                break;
            } else {                        // Reached the end of this dimension
                coord[j] = 0;               // Reset coordinate
            }                               // Loop then continues to increment the next dimension
        }
    }
}

