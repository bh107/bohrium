{
    va_list list;
    va_start(list,tool);
    int64_t nelements = va_arg(list, int64_t);
    {{#OPERAND}}
    {{TYPE}} *a{{NR}}_data = va_arg(list, {{TYPE}}*);
    {{/OPERAND}}
    va_end(list);

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
                                                // Fill up the array
        for(int64_t i=work_offset; i<work_end; ++i) {
            {{#OPERAND}}
            a{{NR}}_data[i] = i;
            {{/OPERAND}}
        }
    }
}

