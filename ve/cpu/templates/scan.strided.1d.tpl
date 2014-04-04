//
// Scan operation on one-dimensional arrays with strided access.
{
    {{#OPERAND}}{{#ARRAY}}
    {{TYPE}} *a{{NR}}_current = a{{NR}}_first;
    {{/ARRAY}}{{/OPERAND}}

    {{TYPE_AXIS}} axis  = *a{{NR_SINPUT}}_first;

    int64_t nelements = a{{NR_FINPUT}}_shape[axis];
    int mthreads = omp_get_max_threads();
    int64_t nworkers = nelements > mthreads ? mthreads : 1;

    {{TYPE_INPUT}} state = ({{TYPE_INPUT}}){{NEUTRAL_ELEMENT}};
    for(int64_t j=0; j<a{{NR_OUTPUT}}_shape[axis]; ++j) {
        {{#OPERATORS}}
        {{OPERATOR}};
        {{/OPERATORS}}

        {{#OPERAND}}{{#ARRAY}}
        a{{NR}}_current += a{{NR}}_stride[axis];
        {{/ARRAY}}{{/OPERAND}}

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

