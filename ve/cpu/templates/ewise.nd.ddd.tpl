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

    int64_t nelements = shape[0];
    for(int64_t i=1; i<ndim; ++i) {
        nelements *= shape[i];
    }

    {{#OPERAND}}{{#ARRAY}}
    assert(a{{NR}}_first != NULL);
    a{{NR}}_first += a{{NR}}_start;
    {{/ARRAY}}{{/OPERAND}}

    int mthreads     = omp_get_max_threads();
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

        {{#OPERAND}}
        {{TYPE}} *a{{NR}}_current = a{{NR}}_first{{#ARRAY}} + work_offset{{/ARRAY}};
        {{/OPERAND}}

        for (int64_t i = work_offset; i<work_end; ++i) {
            {{OPERATOR}};

            {{#OPERAND}}{{#ARRAY}}
            ++a{{NR}}_current;
            {{/ARRAY}}{{/OPERAND}}
        }
    }
}

