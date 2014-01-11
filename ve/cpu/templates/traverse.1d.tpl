/**

    int64_t *shape,
    int64_t ndim,
    [operand] -> [


*/
void {{SYMBOL}}(int tool, ...)
{
    va_list list;               // Unpack arguments
    va_start(list, tool);

    int64_t *shape      = va_arg(list, int64_t*);
    int64_t ndim        = va_arg(list, int64_t);

    {{#OPERAND}}
    {{TYPE}} *a{{NR}}_first   = va_arg(list, {{TYPE}}*);
    {{#ARRAY}}
    int64_t  a{{NR}}_start   = va_arg(list, int64_t);
    int64_t *a{{NR}}_stride  = va_arg(list, int64_t*);
    {{/ARRAY}}
    {{/OPERAND}}

    va_end(list);

    int64_t last_dim    = ndim-1,
            nelements   = shape[last_dim];

    {{#OPERAND}}{{#ARRAY}}
    assert(a{{NR}}_first != NULL);
    a{{NR}}_first += a{{NR}}_start;
    int64_t a{{NR}}_stride_ld = a{{NR}}_stride[last_dim];
    {{/ARRAY}}{{/OPERAND}}

    int mthreads = omp_get_max_threads();
    int64_t nworkers = nelements > mthreads ? mthreads : 1;

    #pragma omp parallel num_threads(nworkers)
    {
        int tid      = omp_get_thread_num();    // Work partitioning
        int nthreads = omp_get_num_threads();

        int64_t work = nelements / nthreads;
        int64_t work_offset = work * tid;
        if (tid==nthreads-1) {
            work += nelements % nthreads;
        }
        int64_t work_end = work_offset+work;
                                                // Pointer fixes
        {{#OPERAND}}
        {{TYPE}} *a{{NR}}_current = a{{NR}}_first{{#ARRAY}} + (work_offset *a{{NR}}_stride_ld){{/ARRAY}};
        {{/OPERAND}}

        for (int64_t i = work_offset; i < work_end; ++i) {
            {{OPERATOR}};
        
            {{#OPERAND}}{{#ARRAY}}
            a{{NR}}_current += a{{NR}}_stride_ld;
            {{/ARRAY}}{{/OPERAND}}
        }
    }
}

