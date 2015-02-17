// Elementwise operation on contigous arrays of any dimension/rank.
//  * Collapses the loops for every dimension into a single loop.
//  * Simplified array-walking (++)
//  * TODO: Vectorization, alias/restrict
{
    const int64_t nelements = iterspace->nelem;
    const int mthreads      = omp_get_max_threads();
    const int64_t nworkers  = nelements > mthreads ? mthreads : 1;
    const int64_t work_split= nelements / nworkers;
    const int64_t work_spill= nelements % nworkers;

    #pragma omp parallel num_threads(nworkers)
    {
        const int tid      = omp_get_thread_num();  // Thread info
        const int nthreads = omp_get_num_threads();

        int64_t work=0, work_offset=0, work_end=0;  // Work distribution
        if (tid < work_spill) {
            work = work_split + 1;
            work_offset = tid * work;
        } else {
            work = work_split;
            work_offset = tid * work + work_spill;
        }
        work_end = work_offset+work;
    
        //        
        // Operand declaration(s)
        {{WALKER_DECLARATION}}

        //
        // Operand offsets(s)
        {{WALKER_OFFSET}}

        for (int64_t eidx = work_offset; eidx<work_end; ++eidx) {
            // Apply operator(s)
            {{OPERATIONS}}
            
            // Increment operands
            {{WALKER_STEP}}
        }
    }
    // TODO: Handle write-out of non-temp and non-const scalars.
}

