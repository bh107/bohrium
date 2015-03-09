//
// Codegen template is used for:
//
//	* MAP|ZIP|GENERATE|FLOOD on 1D strided arrays.
//	* MAP|ZIP|GENERATE|FLOOD on contigous arrays of any dimension/rank.
//
//  * TODO: Vectorization, alias/restrict
{
    const int mthreads = omp_get_max_threads();
    const int64_t nworkers = iterspace->nelem > mthreads ? mthreads : 1;
    const int64_t work_split= iterspace->nelem / nworkers;
    const int64_t work_spill= iterspace->nelem % nworkers;

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

        {{PRAGMA_SIMD}}
        for (int64_t eidx = work_offset; eidx<work_end; ++eidx) {
            // Apply operator(s)
            {{OPERATIONS}}
            
            // Increment operands
            {{WALKER_STEP_LD}}
        }}
    }
    // TODO: Handle write-out of non-temp and non-const scalars.
}

