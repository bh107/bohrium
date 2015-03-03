//
// Codegen template is used for:
//
//	* MAP|ZIP|GENERATE|FLOOD on STRIDED arrays of any dimension/rank.
//
//	Partitions work into chunks of size equal to the inner-most dimension.
//	Distribites work staticly/evenly among threads.
//
{
    const int64_t last_dim  = iterspace->ndim-1;

    int64_t weight[CPU_MAXDIM]; // Helper for step-calculation
    int acc = 1;
    for(int idx=last_dim; idx >=0; --idx) {
        weight[idx] = acc;
        acc *= iterspace->shape[idx];
    }

    const int mthreads          = omp_get_max_threads();
    const int64_t chunksize     = iterspace->shape[last_dim];
    const int64_t nchunks       = iterspace->nelem / chunksize;
    const int64_t nworkers      = nchunks > mthreads ? mthreads : 1;
    const int64_t work_split    = nchunks / nworkers;
    const int64_t work_spill    = nchunks % nworkers;
    
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
        // Stepsize
        {{WALKER_STEPSIZE}}
        const int64_t eidx_begin = work_offset*chunksize;
        const int64_t eidx_end   = work_end*chunksize;
        for(int64_t eidx=eidx_begin; eidx<eidx_end; eidx+=chunksize) {
            // Operand declaration(s)
            {{WALKER_DECLARATION}}
            
            // Walker step outer dimensions
            for (int64_t dim=0; dim < last_dim; ++dim) {    // offset from coord
                const int64_t coord = (eidx / weight[dim]) % iterspace->shape[dim];
                {{WALKER_STEP_OUTER}}
            }

            for (int64_t iidx=0; iidx < chunksize; iidx++) { // Execute array-operations
                {{OPERATIONS}}

                // Walker step innermost dimension
                {{WALKER_STEP_INNER}}
            }
        }}
        // TODO: Handle write-out of non-temp and non-const scalars.
    }
}

