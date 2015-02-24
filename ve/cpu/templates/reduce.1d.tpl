//
// 1D Reduction
{
    const int mthreads = omp_get_max_threads();
    const int64_t nworkers = iterspace->shape[0] > mthreads ? mthreads : 1;
    const int64_t work_split= iterspace->shape[0] / nworkers;
    const int64_t work_spill= iterspace->shape[0] % nworkers;

    {{ETYPE}} partials[nworkers];

    // Parallel reduction, accumulate reduction into partials
    #pragma omp parallel num_threads(nworkers)
    {
        const int tid      = omp_get_thread_num();
        const int nthreads = omp_get_num_threads();

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

        // Axis is ignored for 1D, since there is only one...
        {{ETYPE}} accu = {{NEUTRAL_ELEMENT}};
        for (int64_t eidx = work_offset; eidx<work_end; ++eidx) {
            // Apply operator(s)
            {{PAR_OPERATIONS}}
            
            // Increment operands
            {{WALKER_STEP_LD}}
        }

        // Write out the result to partials
        partials[tid] = accu;
        }
    }
   
    // Sequential reduction, accumulate partials into scalar
    {{ETYPE}} accu = partials[0];
    for(int64_t pidx=1; pidx<nworkers; ++pidx) {
        {{SEQ_OPERATIONS}}
    }
    // Write it out to memory

}

