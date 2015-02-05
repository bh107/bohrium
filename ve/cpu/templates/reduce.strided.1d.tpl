//
// Reduction of on one-dimensional arrays using strided indexing
{
    {{#OPERAND}}
    {{#SCALAR}}{{TYPE}} a{{NR}}_current = *a{{NR}}_first;{{/SCALAR}}
    {{#SCALAR_CONST}}const {{TYPE}} a{{NR}}_current = *a{{NR}}_first;{{/SCALAR_CONST}}
    {{#SCALAR_TEMP}}{{TYPE}} a{{NR}}_current;{{/SCALAR_TEMP}}
    {{#ARRAY}}{{TYPE}}* a{{NR}}_current = a{{NR}}_first;{{/ARRAY}}
    {{/OPERAND}}

    {{TYPE_AXIS}} axis = *a{{NR_SINPUT}}_first;
    {{TYPE_INPUT}} state = 0;

    const int64_t nelements   = iterspace->shape[axis];
    const int mthreads        = omp_get_max_threads();
    const int64_t nworkers    = nelements > mthreads ? mthreads : 1;

    #pragma omp parallel for reduction(+:state) num_threads(nworkers)
    for(int64_t j=0; j<iterspace->shape[axis]; ++j) {
        {{TYPE_INPUT}} *tmp_current = a{{NR_FINPUT}}_current + a{{NR_FINPUT}}_stride[axis]*j;

        {{#OPERATORS}}
        {{OPERATOR}};
        {{/OPERATORS}}
    }
    *a{{NR_OUTPUT}}_current = state;

    // TODO: Handle write-out of non-temp and non-const scalars.
}

