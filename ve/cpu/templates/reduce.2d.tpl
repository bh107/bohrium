//
// Reduction on two-dimensional arrays using strided indexing
{
    {{ATYPE}} axis = *{{OPD_IN2}}_first;
    {{ATYPE}} other_axis = (axis==0) ? 1 : 0;

    const int64_t nelements   = iterspace->shape[other_axis];
    const int mthreads        = omp_get_max_threads();
    const int64_t nworkers    = nelements > mthreads ? mthreads : 1;

    #pragma omp parallel for num_threads(nworkers)
    for(int64_t other_idx=0; other_idx<iterspace->shape[other_axis]; ++other_idx) {
       
        {{ETYPE}}* {{OPD_IN1}} = {{OPD_IN1}}_first + \
                                 {{OPD_IN1}}_stride[other_axis] *other_idx;
        {{ETYPE}} accu = {{NEUTRAL_ELEMENT}};
        for(int64_t axis_idx=0; axis_idx<iterspace->shape[axis]; ++axis_idx) { // Accumulate
            {{PAR_OPERATIONS}}
            {{OPD_IN1}} += {{OPD_IN1}}_stride[axis];
        }
        *({{OPD_OUT}}_first + {{OPD_OUT}}_stride[0]*other_idx) = accu;
    }
}

