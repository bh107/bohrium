//
// Reduction on two-dimensional arrays using strided indexing
{
    {{ATYPE}} axis = *{{OPD_IN2}}_first;
    {{ATYPE}} other_axis = (axis==0) ? 1 : 0;

    const int64_t nelements   = iterspace->shape[other_axis];
    const int mthreads        = omp_get_max_threads();
    const int64_t nworkers    = nelements > mthreads ? mthreads : 1;

    #pragma omp parallel for num_threads(nworkers)
    for(int64_t j=0; j<iterspace->shape[other_axis]; ++j) {
        // todo: need another step function, stride-step
        {{ETYPE}}* {{OPD_IN1}} = {{OPD_IN1}}_first + j;

        {{ETYPE}} accu = {{NEUTRAL_ELEMENT}};             // Scalar-temp 
        for(int64_t i=0; i<iterspace->shape[axis]; ++i) { // Accumulate
            // todo: need another step function, stride-step
            ++{{OPD_IN1}};
            {{PAR_OPERATIONS}}
        }
        // Update array
        // todo: need another step function, stride-step
        *({{OPD_OUT}}_first + j) = accu; 
    }

    // TODO: Handle write-out of non-temp and non-const scalars.
}

