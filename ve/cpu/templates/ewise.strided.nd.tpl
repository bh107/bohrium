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
    {{#ARRAY}}int64_t  a{{NR}}_stride_ld = a{{NR}}_stride[last_dim];{{/ARRAY}}
    {{/OPERAND}}

    int64_t weight[CPU_MAXDIM];
    int acc = 1;
    for(int idx=iterspace->ndim-1; idx >=0; --idx) {
        weight[idx] = acc;
        acc *= iterspace->shape[idx];
    }

    #pragma omp parallel for schedule(static)
    for(int64_t eidx=0; eidx<nelements; ++eidx) {

        {{#OPERAND}}{{#ARRAY}}
        {{TYPE}}* a{{NR}}_current = a{{NR}}_first;
        {{/ARRAY}}{{/OPERAND}}
        
        for (int64_t dim=0; dim <= last_dim; ++dim) {
            const int64_t dim_position = (eidx / weight[dim]) % iterspace->shape[dim];
            {{#OPERAND}}{{#ARRAY}}
            a{{NR}}_current += dim_position * a{{NR}}_stride[dim];
            {{/ARRAY}}{{/OPERAND}}
        }

        {{#OPERATORS}}
        {{OPERATOR}};
        {{/OPERATORS}}
    }
    // TODO: Handle write-out of non-temp and non-const scalars.
}

