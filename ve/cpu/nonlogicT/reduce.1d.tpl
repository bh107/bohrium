//
// Reduction of on one-dimensional arrays using strided indexing
{
    {{WALKER_DECLARATION}}

    {{TYPE_AXIS}} axis = *a{{NR_SINPUT}}_first;
    {{TYPE_INPUT}} state = 0;

    const int64_t nelements   = iterspace->shape[axis];
    const int mthreads        = omp_get_max_threads();
    const int64_t nworkers    = nelements > mthreads ? mthreads : 1;

    #pragma omp parallel for reduction(+:state) num_threads(nworkers)
    for(int64_t j=0; j<iterspace->shape[axis]; ++j) {
        {{TYPE_INPUT}} *tmp_current = a{{NR_FINPUT}}_current + a{{NR_FINPUT}}_stride[axis]*j;
        
        {{OPERATIONS}}
    }
    *a{{NR_OUTPUT}}_current = state;

    // TODO: Handle write-out of non-temp and non-const scalars.
}

