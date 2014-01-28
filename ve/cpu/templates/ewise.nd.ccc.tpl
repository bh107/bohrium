{
    {{#OPERAND}}
    {{TYPE}} *a{{NR}}_first = args->data[{{NR}}];
    {{/OPERAND}}

    {{#OPERAND}}{{#ARRAY}}
    assert(a{{NR}}_first != NULL);
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
            {{#LOOP_BODY}}
            {{OPERATOR}};
            {{/LOOP_BODY}}

            {{#OPERAND}}{{#ARRAY}}
            ++a{{NR}}_current;
            {{/ARRAY}}{{/OPERAND}}
        }
    }
}

