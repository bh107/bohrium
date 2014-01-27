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

    int64_t ld  = ndim-1,   // Traversal variables
            sld = ndim-2,
            nelements = shape[sld];

    {{#OPERAND}}{{#ARRAY}}
    int64_t a{{NR}}_stride_ld    = a{{NR}}_stride[ld];
    int64_t a{{NR}}_stride_sld   = a{{NR}}_stride[sld];
    int64_t a{{NR}}_rewind_ld = shape[ld]*a{{NR}}_stride_ld;
    a{{NR}}_first += a{{NR}}_start;
    {{/ARRAY}}{{/OPERAND}}

    int mthreads = omp_get_max_threads();
    int64_t nworkers = nelements > mthreads ? mthreads : 1;

    #pragma omp parallel num_threads(nworkers)
    {
        int tid      = omp_get_thread_num();    // Work partitioning
        int nthreads = omp_get_num_threads();

        int64_t work = shape[sld] / nthreads;
        int64_t work_offset = work * tid;
        if (tid==nthreads-1) {
            work += shape[sld] % nthreads;
        }
        int64_t work_end = work_offset+work;
                                                // Pointer fixes
        {{#OPERAND}}
        {{TYPE}} *a{{NR}}_current = a{{NR}}_first{{#ARRAY}} + (work_offset * a{{NR}}_stride_sld){{/ARRAY}};
        {{/OPERAND}}

        for(int64_t j=work_offset; j<work_end; ++j) {
            for (int64_t i = 0; i < shape[ld]; ++i) {
                {{#LOOP_BODY}}
                {{OPERATOR}};
                {{/LOOP_BODY}}
               
                {{#OPERAND}}{{#ARRAY}}
                a{{NR}}_current += a{{NR}}_stride_ld;
                {{/ARRAY}}{{/OPERAND}}
            }
            {{#OPERAND}}{{#ARRAY}}
            a{{NR}}_current -= a{{NR}}_rewind_ld;
            a{{NR}}_current += a{{NR}}_stride_sld;
            {{/ARRAY}}{{/OPERAND}}
        }
    }
}

