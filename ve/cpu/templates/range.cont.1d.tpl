//
// Unclassified operation... iota/range/generator of data..
{
    int64_t nelements = a{{NR_OUTPUT}}_nelem;

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

        {{#OPERAND}}{{#ARRAY}}
        {{TYPE}} *a{{NR}}_current = a{{NR}}_first;
        {{/ARRAY}}{{/OPERAND}}
                                                // Fill up the array
        for(int64_t i=work_offset; i<work_end; ++i) {
            {{#OPERAND}}
            a{{NR}}_current[i] = i;
            {{/OPERAND}}
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

