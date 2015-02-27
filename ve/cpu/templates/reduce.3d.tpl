//
// Reduction on three-dimensional arrays using strided indexing
{
#define OUTER 1
#define INNER 0

    const {{ATYPE}} axis = *{{OPD_IN2}}_first;

    {{ATYPE}} outer_axis;
    {{ATYPE}} inner_axis;
    if (axis == 0) {
        outer_axis = 2;
        inner_axis = 1;
    } else if (axis==1) {
        outer_axis = 2;
        inner_axis = 0;
    } else if (axis==2) {
        outer_axis = 1;
        inner_axis = 0;
    }
    
    const int64_t nelements   = iterspace->shape[OUTER] + iterspace->shape[INNER];
    const int mthreads        = omp_get_max_threads();
    const int64_t nworkers    = nelements > mthreads ? mthreads : 1;

    #pragma omp parallel for num_threads(nworkers) collapse(2)
    for(int64_t outer_idx=0; outer_idx<iterspace->shape[outer_axis]; ++outer_idx) {
        for(int64_t inner_idx=0; inner_idx<iterspace->shape[inner_axis]; ++inner_idx) {
            // todo: need another step function, stride-step
            {{ETYPE}}* {{OPD_IN1}} = {{OPD_IN1}}_first +\
                          outer_idx* {{OPD_IN1}}_stride[outer_axis] +\
                          inner_idx* {{OPD_IN1}}_stride[inner_axis];

            //{{ETYPE}} accu = {{NEUTRAL_ELEMENT}};
            {{ETYPE}} accu = *{{OPD_IN1}};
            for(int64_t axis_idx=1; axis_idx<iterspace->shape[axis]; ++axis_idx) {
                {{OPD_IN1}} += {{OPD_IN1}}_stride[axis];
                {{PAR_OPERATIONS}}
            }
            // todo: need another step function, stride-step
            *({{OPD_OUT}}_first + outer_idx*{{OPD_OUT}}_stride[OUTER] + inner_idx*{{OPD_OUT}}_stride[INNER]) = accu;
        }
    }
}

