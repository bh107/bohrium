//
// Elementwise operation on strided arrays of any dimension/rank.
{
    int64_t nelements = iterspace->nelem;

    int64_t last_dim  = iterspace->ndim-1;
    int64_t shape_ld  = iterspace->shape[last_dim];
    int64_t eidx     = 0;
    int64_t j;

    {{#OPERAND}}
    {{#SCALAR}}{{TYPE}} a{{NR}}_current = *a{{NR}}_first;{{/SCALAR}}
    {{#SCALAR_CONST}}const {{TYPE}} a{{NR}}_current = *a{{NR}}_first;{{/SCALAR_CONST}}
    {{#SCALAR_TEMP}}{{TYPE}} a{{NR}}_current;{{/SCALAR_TEMP}}
    {{#ARRAY}}int64_t  a{{NR}}_stride_ld = a{{NR}}_stride[last_dim];{{/ARRAY}}
    {{/OPERAND}}

    int mthreads = omp_get_max_threads();
    int64_t nworkers = nelements > mthreads ? mthreads : 1;

    int64_t weight[CPU_MAXDIM];
    int acc = 1;
    for(int idx=iterspace->ndim-1; idx >=0; --idx) {
        weight[idx] = acc;
        acc *= iterspace->shape[idx];
    }

    int64_t coord[CPU_MAXDIM];
    memset(coord, 0, CPU_MAXDIM * sizeof(int64_t));

    while (eidx < nelements) {
        // Reset offsets
        {{#OPERAND}}{{#ARRAY}}
        {{TYPE}}* a{{NR}}_current = a{{NR}}_first;
        {{/ARRAY}}{{/OPERAND}}

        for (j=0; j<last_dim; ++j) {           // Compute offset based on coordinate
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
        eidx += shape_ld;

        for (j=0; j < last_dim; ++j) {
            coord[j] = (eidx / weight[j]) % iterspace->shape[j];
        }
    }
    // TODO: Handle write-out of non-temp and non-const scalars.
}

