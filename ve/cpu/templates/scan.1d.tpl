/*
int scan(
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
    int64_t  a0_start   = va_arg(list, int64_t);    // Scan result
    int64_t *a0_stride  = va_arg(list, int64_t*);
    int64_t *a0_shape   = va_arg(list, int64_t*);
    int64_t  a0_ndim    = va_arg(list, int64_t);

    {{TYPE_A1}} *a1_first    = va_arg(list, {{TYPE_A1}}*);
    int64_t  a1_start   = va_arg(list, int64_t);    // Input to scan
    int64_t *a1_stride  = va_arg(list, int64_t*);
    int64_t *a1_shape   = va_arg(list, int64_t*);
    int64_t  a1_ndim    = va_arg(list, int64_t);

    int64_t axis = va_arg(list, int64_t);           // Axis to scan

    va_end(list);                                   // **DONE UNPACKING**

    {{TYPE_A0}} *a0_current = a0_first + a0_start;  // Ptr to first output elem
    {{TYPE_A1}} *a1_current = a1_first + a1_start;  // Ptr to first input elem
    {{TYPE_A1}} cvar = *a0_current;                 // Carry the accumulated

    int64_t nelements = a1_shape[axis];
    int mthreads = omp_get_max_threads();
    int64_t nworkers = nelements > mthreads ? mthreads : 1;

    for(int64_t j=0; j<a0_shape[axis]; ++j) {
        {{OPERATOR}};
        a0_current += a0_stride[axis];
        a1_current += a1_stride[axis];
    }
    
    return 1;
}

