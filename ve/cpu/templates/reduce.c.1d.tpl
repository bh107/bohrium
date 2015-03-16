//
// Codegen template is used for:
//
//  * REDUCE COMPLETE on ND arrays of CONTIGUOUS LAYOUT.
//  * REDUCE COMPLETE on 1D arrays of ANY LAYOUT.
//
{
    const int mthreads = omp_get_max_threads();
    const int64_t nworkers = iterspace->nelem > mthreads ? mthreads : 1;
    const int64_t work_split= iterspace->nelem / nworkers;
    const int64_t work_spill= iterspace->nelem % nworkers;

    *({{OPD_OUT}}_first) = {{NEUTRAL_ELEMENT}};

    #pragma omp parallel num_threads(nworkers)
    {
        const int tid = omp_get_thread_num();

        int64_t work=0, work_offset=0, work_end=0;
        if (tid < work_spill) {
            work = work_split + 1;
            work_offset = tid * work;
        } else {
            work = work_split;
            work_offset = tid * work + work_spill;
        }
        work_end = work_offset+work;

        if (work) {
        // Operand declaration(s)
        {{WALKER_DECLARATION}}
        // Operand offsets(s)
        {{WALKER_OFFSET}}
        // Stepsize
        {{WALKER_STEPSIZE}}

        {{ETYPE}} accu = *({{OPD_IN1}});
        {{PRAGMA_SIMD}}
        for (int64_t eidx = work_offset+1; eidx<work_end; ++eidx) {
            // Increment operands
            {{WALKER_STEP_LD}}

            // Apply operator(s)
            {{REDUCE_OPER}}
        }
        {{REDUCE_SYNC}}
        {{REDUCE_OPER_COMBINE}}
        }
    }
}
