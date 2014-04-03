// Elementwise operation on contigous arrays of any dimension/rank.
//  * Collapses the loops for every dimension into a single loop.
//  * Simplified array-walking (++)
//  * TODO: Vectorization, alias/restrict
{
    int64_t nelements = 1;
    for(int k=0; k<a{{NR_OUTPUT}}_ndim; ++k) {
        nelements *= a{{NR_OUTPUT}}_shape[k];
    }

    int mthreads      = omp_get_max_threads();
    int64_t nworkers  = nelements > mthreads ? mthreads : 1;

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
        {{TYPE}} *a{{NR}}_current = a{{NR}}_first + work_offset;
        {{/ARRAY}}{{/OPERAND}}

        for (int64_t i = work_offset; i<work_end; ++i) {
            {{#OPERATORS}}
            {{OPERATOR}};
            {{/OPERATORS}}

            {{#OPERAND}}{{#ARRAY}}
            ++a{{NR}}_current;
            {{/ARRAY}}{{/OPERAND}}
        }
    }

    {{#OPERAND}}{{#SCALAR}}
    // TODO: Write scalars out to main-memory
    {{/SCALAR}}{{/OPERAND}}
}

