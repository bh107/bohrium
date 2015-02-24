//
// Elementwise operation on one-dimensional arrays using strided indexing
{
    const int64_t nelements = iterspace->nelem;
    const int mthreads      = omp_get_max_threads();
    const int64_t nworkers  = nelements > mthreads ? mthreads : 1;
    const int64_t work_split= nelements / nworkers;
    const int64_t work_spill= nelements % nworkers;

    #pragma omp parallel num_threads(nworkers)
    {
        const int tid      = omp_get_thread_num();  // Thread info
        const int nthreads = omp_get_num_threads();

        int64_t work=0, work_offset=0, work_end=0;  // Work distribution
        if (tid < work_spill) {
            work = work_split + 1;
            work_offset = tid * work;
        } else {
            work = work_split;
            work_offset = tid * work + work_spill;
        }
        work_end = work_offset+work;
        if (work) {
        // Step
        {{#OPERAND}}{{#ARRAY}}
        const int64_t a{{NR}}_step_ld = a{{NR}}_stride[0];
        {{/ARRAY}}{{/OPERAND}}

        // Operands
        {{#OPERAND}}
        {{#SCALAR}}{{TYPE}} a{{NR}}_current = *a{{NR}}_first;{{/SCALAR}}
        {{#SCALAR_CONST}}const {{TYPE}} a{{NR}}_current = *a{{NR}}_first;{{/SCALAR_CONST}}
        {{#SCALAR_TEMP}}{{TYPE}} a{{NR}}_current;{{/SCALAR_TEMP}}
        {{#ARRAY}}{{TYPE}}* a{{NR}}_current = a{{NR}}_first;{{/ARRAY}}
        {{/OPERAND}}

        // Do offset
        {{#OPERAND}}{{#ARRAY}}
        a{{NR}}_current += work_offset * a{{NR}}_stride[0];
        {{/ARRAY}}{{/OPERAND}}

        for (int64_t eidx = work_offset; eidx < work_end; ++eidx) {
            {{#OPERATORS}}
            {{OPERATOR}};
            {{/OPERATORS}}
        
            {{#OPERAND}}{{#ARRAY}}
            a{{NR}}_current += a{{NR}}_step_ld;
            {{/ARRAY}}{{/OPERAND}}
        }}
    }
    // TODO: Handle write-out of non-temp and non-const scalars.
}
