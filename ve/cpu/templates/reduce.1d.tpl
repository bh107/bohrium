//
// Codegen template is used for:
//
//  * REDUCE on 1D strided and contiguous arrays.
//
{
    const {{ATYPE}} axis = *{{OPD_IN2}}_first;

    const int64_t nelements   = iterspace->shape[axis];
    const int mthreads        = omp_get_max_threads();
    const int64_t nworkers    = nelements > mthreads ? mthreads : 1;

    {{ETYPE}} accu = {{NEUTRAL_ELEMENT}};
    #pragma omp parallel for reduction(+:accu) num_threads(nworkers)
    for(int64_t eidx=0; eidx<iterspace->shape[axis]; ++eidx) {
        {{ETYPE}}* {{OPD_IN1}} = {{OPD_IN1}}_first + {{OPD_IN1}}_stride[axis]*eidx;

        {{PAR_OPERATIONS}}
    }
    *{{OPD_OUT}}_first = accu;

}

