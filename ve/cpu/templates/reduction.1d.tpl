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

    int64_t axis
)
*/
int {{SYMBOL}}(int tool, ...)
{
    va_list list;                                   // **UNPACK PARAMETERS**
    va_start(list, tool);

    {{TYPE_A0}} *a0_first   = va_arg(list, {{TYPE_A0}}*);
    int64_t  a0_start   = va_arg(list, int64_t);    // Reduction result
    int64_t *a0_stride  = va_arg(list, int64_t*);
    int64_t *a0_shape   = va_arg(list, int64_t*);
    int64_t  a0_ndim    = va_arg(list, int64_t);

    {{TYPE_A1}} *a1_first    = va_arg(list, {{TYPE_A1}}*);
    int64_t  a1_start   = va_arg(list, int64_t);    // Input to reduce
    int64_t *a1_stride  = va_arg(list, int64_t*);
    int64_t *a1_shape   = va_arg(list, int64_t*);
    int64_t  a1_ndim    = va_arg(list, int64_t);

    int64_t axis = va_arg(list, int64_t);           // Reduction axis

    va_end(list);                                   // **DONE UNPACKING**

    {{TYPE_A0}} *a0_current = a0_first + a0_start;  // Point to first element in output.
    {{TYPE_A1}} *a1_current = a1_first + a1_start;  // Point to first element in input.
    {{TYPE_A1}} rvar = 0;                           // Use the first element as temp

    int64_t nelements = a1_shape[axis];
    int mthreads = omp_get_max_threads();
    int64_t nworkers = nelements > mthreads ? mthreads : 1;

    #pragma omp parallel for reduction(+:rvar) num_threads(nworkers)
    for(int64_t j=0; j<a1_shape[axis]; ++j) {
        {{TYPE_A1}} *tmp_current = a1_current + a1_stride[axis]*j;
        {{OPERATOR}};
    }
    *a0_current = rvar;
    
    return 1;
}

