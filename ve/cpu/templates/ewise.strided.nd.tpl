//
// Elementwise operation on strided arrays of any dimension/rank.
{
    int64_t nelements = 1;
    for(int k=0; k<a{{NR_OUTPUT}}_ndim; ++k) {
        nelements *= a{{NR_OUTPUT}}_shape[k];
    }

    int64_t last_dim  = a{{NR_OUTPUT}}_ndim-1;
    int64_t shape_ld  = a{{NR_OUTPUT}}_shape[last_dim];
    int64_t last_e    = nelements-1;
    int64_t cur_e     = 0;
    int64_t j;

    {{#OPERAND}}{{#ARRAY}}
    int64_t  a{{NR}}_stride_ld = a{{NR}}_stride[last_dim];
    {{/ARRAY}}{{/OPERAND}}

    int mthreads = omp_get_max_threads();
    int64_t nworkers = nelements > mthreads ? mthreads : 1;

    int64_t coord[CPU_MAXDIM];
    memset(coord, 0, CPU_MAXDIM * sizeof(int64_t));

    while (cur_e <= last_e) {
        // Reset offsets
        {{#OPERAND}}{{#ARRAY}}
        {{TYPE}}* a{{NR}}_current = a{{NR}}_first;
        {{/ARRAY}}{{/OPERAND}}

        for (j=0; j<=last_dim; ++j) {           // Compute offset based on coordinate
            {{#OPERAND}}{{#ARRAY}}
            a{{NR}}_current += coord[j] * a{{NR}}_stride[j];
            {{/ARRAY}}{{/OPERAND}}
        }

        for (j = 0; j < shape_ld; j++) {        // Iterate over "last" / "innermost" dimension
            {{#OPERATORS}}
            {{OPERATOR}};
            {{/OPERATORS}}

            {{#OPERAND}}{{#ARRAY}}
            a{{NR}}_current += a{{NR}}_stride_ld;
            {{/ARRAY}}{{/OPERAND}}
        }
        cur_e += shape_ld;

        // coord[last_dim] is never used, only all the other coord[dim!=last_dim]
        for (j = last_dim-1; j >= 0; --j) {  // Increment coordinates for the remaining dimensions
            coord[j]++;
            if (coord[j] < a{{NR_OUTPUT}}_shape[j]) {      // Still within this dimension
                break;
            } else {                        // Reached the end of this dimension
                coord[j] = 0;               // Reset coordinate
            }                               // Loop then continues to increment the next dimension
        }
    }
    
    {{#OPERAND}}{{#SCALAR}}
    // Write scalar-operand to main-memory;
    // Note this is only necessary for non-temporary scalar-operands.
    // So this code should only be generated for non-temps.
    if ({{NR_OUTPUT}} == {{NR}}) {
        *a{{NR}}_first = a{{NR}}_current;
    }
    {{/SCALAR}}{{/OPERAND}}
}

