{
#define OUTER 1
#define INNER 0
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
    
    int64_t nelements = a0_shape[OUTER]+a1_shape[INNER];
    int mthreads = omp_get_max_threads();
    int64_t nworkers = nelements > mthreads ? mthreads : 1;

    #pragma omp parallel for num_threads(nworkers) collapse(2)
    for(int64_t i=0; i<a0_shape[OUTER]; ++i) {
        for(int64_t j=0; j<a0_shape[INNER]; ++j) {
            {{TYPE_INPUT}} *tmp_current = a1_first + a1_start + \
                                       i*a1_stride[outer_axis] + \
                                       j*a1_stride[inner_axis];

            {{TYPE_INPUT}} rvar = *tmp_current;
            for(int64_t k=1; k<a1_shape[axis]; ++k) {
                tmp_current += a1_stride[axis];

                {{#LOOP_BODY}}
                {{OPERATOR}};
                {{/LOOP_BODY}}
            }
            *(a0_first + a0_start + i*a0_stride[OUTER] + j*a0_stride[INNER]) = rvar;
        }
    }
}

