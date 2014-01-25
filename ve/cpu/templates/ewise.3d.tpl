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
            tld = ndim-3,
            nelements = shape[tld];

    {{#OPERAND}}{{#ARRAY}}
    assert(a{{NR}}_first != NULL);
    int64_t a{{NR}}_stride_ld    = a{{NR}}_stride[ld];
    int64_t a{{NR}}_stride_sld   = a{{NR}}_stride[sld];
    int64_t a{{NR}}_stride_tld   = a{{NR}}_stride[tld];

    int64_t a{{NR}}_rewind_ld    = shape[ld]*a{{NR}}_stride[ld];
    int64_t a{{NR}}_rewind_sld   = shape[sld]*a{{NR}}_stride[sld];
    a{{NR}}_first += a{{NR}}_start;
    {{/ARRAY}}{{/OPERAND}}

    int mthreads = omp_get_max_threads();
    int64_t nworkers = nelements > mthreads ? mthreads : 1;

    #pragma omp parallel num_threads(nworkers)
    {
        int tid      = omp_get_thread_num();    // Work partitioning
        int nthreads = omp_get_num_threads();

        int64_t work = shape[tld] / nthreads;
        int64_t work_offset = work * tid;
        if (tid==nthreads-1) {
            work += shape[tld] % nthreads;
        }
        int64_t work_end = work_offset+work;
                                                // Pointer fixes
        {{#OPERAND}}
        {{TYPE}} *a{{NR}}_current = a{{NR}}_first{{#ARRAY}} + (work_offset * a{{NR}}_stride_tld){{/ARRAY}};
        {{/OPERAND}}

        for (int64_t k=work_offset; k<work_end; ++k) {
            for (int64_t j = 0; j<shape[sld]; ++j) {
                for (int64_t i = 0; i<shape[ld]; ++i) {
                    {{OPERATOR}};

                    {{#OPERAND}}{{#ARRAY}}
                    a{{NR}}_current += a{{NR}}_stride_ld;
                    {{/ARRAY}}{{/OPERAND}}
                }
                
                {{#OPERAND}}{{#ARRAY}}
                a{{NR}}_current -= a{{NR}}_rewind_ld;
                a{{NR}}_current += a{{NR}}_stride_sld;
                {{/ARRAY}}{{/OPERAND}}
            }
            {{#OPERAND}}{{#ARRAY}}
            a{{NR}}_current -= a{{NR}}_rewind_sld;
            a{{NR}}_current += a{{NR}}_stride_tld;
            {{/ARRAY}}{{/OPERAND}}
        }
    }
}

