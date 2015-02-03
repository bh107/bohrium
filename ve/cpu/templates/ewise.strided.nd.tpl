//
// Elementwise operation on strided arrays of any dimension/rank.
{
    const int64_t nelements = iterspace->nelem;
    const int64_t last_dim  = iterspace->ndim-1;
    const int64_t shape_ld  = iterspace->shape[last_dim];

    {{#OPERAND}}
    {{#SCALAR}}{{TYPE}} a{{NR}}_current = *a{{NR}}_first;{{/SCALAR}}
    {{#SCALAR_CONST}}const {{TYPE}} a{{NR}}_current = *a{{NR}}_first;{{/SCALAR_CONST}}
    {{#SCALAR_TEMP}}{{TYPE}} a{{NR}}_current;{{/SCALAR_TEMP}}
    {{#ARRAY}}int64_t a{{NR}}_stride_ld = a{{NR}}_stride[last_dim];{{/ARRAY}}
    {{/OPERAND}}

    int64_t weight[CPU_MAXDIM];
    int acc = 1;
    for(int idx=iterspace->ndim-1; idx >=0; --idx) {
        weight[idx] = acc;
        acc *= iterspace->shape[idx];
    }

    #pragma omp parallel for schedule(static)
    for(int64_t eidx=0; eidx<nelements; eidx+=shape_ld) {

        {{#OPERAND}}{{#ARRAY}}
        {{TYPE}}* a{{NR}}_current = a{{NR}}_first;
        {{/ARRAY}}{{/OPERAND}}
        for (int64_t dim=0; dim < last_dim; ++dim) {    // offset from coord
            const int64_t coord = (eidx / weight[dim]) % iterspace->shape[dim];
            {{#OPERAND}}{{#ARRAY}}
            a{{NR}}_current += coord * a{{NR}}_stride[dim];
            {{/ARRAY}}{{/OPERAND}}
        }

        for (int64_t iidx=0; iidx < shape_ld; iidx++) { // Execute array-operations
            {{#OPERATORS}}
            {{OPERATOR}};
            {{/OPERATORS}}

            {{#OPERAND}}{{#ARRAY}}
            a{{NR}}_current += a{{NR}}_stride_ld;
            {{/ARRAY}}{{/OPERAND}}
        }
    }
    // TODO: Handle write-out of non-temp and non-const scalars.
}

