//
// Scan operation on one-dimensional arrays with strided access.
{
    {{#OPERAND}}
    {{#SCALAR}}{{TYPE}} a{{NR}}_current = *a{{NR}}_first;{{/SCALAR}}
    {{#SCALAR_CONST}}const {{TYPE}} a{{NR}}_current = *a{{NR}}_first;{{/SCALAR_CONST}}
    {{#SCALAR_TEMP}}{{TYPE}} a{{NR}}_current;{{/SCALAR_TEMP}}
    {{#ARRAY}}{{TYPE}}* a{{NR}}_current = a{{NR}}_first;{{/ARRAY}}
    {{/OPERAND}}

    {{TYPE_AXIS}} axis = *a{{NR_SINPUT}}_first;

    const int64_t nelements = iterspace->shape[axis];
    const int mthreads = omp_get_max_threads();
    const int64_t nworkers = nelements > mthreads ? mthreads : 1;

    {{TYPE_INPUT}} state = ({{TYPE_INPUT}}){{NEUTRAL_ELEMENT}};
    for(int64_t j=0; j<iterspace->shape[axis]; ++j) {
        {{#OPERATORS}}
        {{OPERATOR}};
        {{/OPERATORS}}

        {{#OPERAND}}{{#ARRAY}}
        a{{NR}}_current += a{{NR}}_stride[axis];
        {{/ARRAY}}{{/OPERAND}}

    }

    // TODO: Handle write-out of non-temp and non-const scalars.
}

