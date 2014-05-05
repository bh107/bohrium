//
// Reduction on three-dimensional arrays using strided indexing
{
#define OUTER 1
#define INNER 0

    {{#OPERAND}}{{#ARRAY}}
    {{TYPE}} *a{{NR}}_current = a{{NR}}_first + a{{NR}}_start;
    {{/ARRAY}}{{/OPERAND}}

    {{TYPE_AXIS}} axis = *a{{NR_SINPUT}}_first;

    {{TYPE_AXIS}} outer_axis;
    {{TYPE_AXIS}} inner_axis;
    if (axis == 0) {
        outer_axis = 2;
        inner_axis = 1;
    } else if (axis==1) {
        outer_axis = 2;
        inner_axis = 0;
    } else if (axis==2) {
        outer_axis = 1;
        inner_axis = 0;
    }
    
    int64_t nelements   = a{{NR_FINPUT}}_shape[OUTER]+a{{NR_FINPUT}}_shape[INNER];
    int mthreads        = omp_get_max_threads();
    int64_t nworkers    = nelements > mthreads ? mthreads : 1;

    #pragma omp parallel for num_threads(nworkers) collapse(2)
    for(int64_t i=0; i<a{{NR_OUTPUT}}_shape[OUTER]; ++i) {
        for(int64_t j=0; j<a{{NR_OUTPUT}}_shape[INNER]; ++j) {
            {{TYPE_INPUT}} *tmp_current = a{{NR_FINPUT}}_first + a{{NR_FINPUT}}_start + \
                                        i*a{{NR_FINPUT}}_stride[outer_axis] + \
                                        j*a{{NR_FINPUT}}_stride[inner_axis];

            {{TYPE_INPUT}} state = *tmp_current;
            for(int64_t k=1; k<a{{NR_FINPUT}}_shape[axis]; ++k) {
                tmp_current += a{{NR_FINPUT}}_stride[axis];

                {{#OPERATORS}}
                {{OPERATOR}};
                {{/OPERATORS}}
            }
            *(a{{NR_OUTPUT}}_first + a{{NR_OUTPUT}}_start + i*a{{NR_OUTPUT}}_stride[OUTER] + j*a{{NR_OUTPUT}}_stride[INNER]) = state;
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

