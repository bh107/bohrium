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
int {{SYMBOL}}(int tool, ...)
{
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
    {{TYPE_INPUT}} rvar = 0;                        // Use the first element as temp

    int64_t nelements = a1_shape[axis];
    int mthreads = omp_get_max_threads();
    int64_t nworkers = nelements > mthreads ? mthreads : 1;

    #pragma omp parallel for reduction(+:rvar) num_threads(nworkers)
    for(int64_t j=0; j<a1_shape[axis]; ++j) {
        {{TYPE_INPUT}} *tmp_current = a1_current + a1_stride[axis]*j;

        {{OPERATOR}};
    }
    *a0_current = rvar;
    
    return 1;
}

