//
//	Walks the iteration-space using outer/inner loop constructs.
//
//	* MAP|ZIP|GENERATE|REDUCE_COMPLETE on STRIDED arrays of any dimension/rank.
//
//	Partitions work into chunks of size equal to the inner-most dimension.
//
{
    const int64_t inner_dim  = iterspace->ndim-1;

    int64_t weight[CPU_MAXDIM]; // Helper for step-calculation
    int64_t acc = 1;
    for(int64_t idx=inner_dim; idx >=0; --idx) {
        weight[idx] = acc;
        acc *= iterspace->shape[idx];
    }

    const int mthreads          = omp_get_max_threads();
    const int64_t chunksize     = iterspace->shape[inner_dim];
    const int64_t nchunks       = iterspace->nelem / chunksize;
    const int64_t nworkers      = nchunks > mthreads ? mthreads : 1;
    const int64_t work_split    = nchunks / nworkers;
    const int64_t work_spill    = nchunks % nworkers;
   
    // Initialize accumulated operand
    {{ACCU_OPD_INIT}}
 
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
        work_end = work_offset + work;

        if (work) {
        // Walker STRIDE_INNER - begin
        {{WALKER_STRIDE_INNER}}
        // Walker STRIDE_INNER - end

        // Accumulator DECLARE - begin
        {{ACCU_LOCAL_DECLARE}}
        // Accumulator DECLARE - end        

        // Iteration space
        const int64_t eidx_begin = work_offset*chunksize;
        const int64_t eidx_end   = work_end*chunksize;
        for(int64_t eidx=eidx_begin; eidx<eidx_end; eidx+=chunksize) {
            // Walker declaration(s) - begin
            {{WALKER_DECLARATION}}
            // Walker declaration(s) - end

            // Walker step OUTER / operand offset - begin
            for (int64_t dim=0; dim < inner_dim; ++dim) {
                const int64_t coord = (eidx / weight[dim]) % iterspace->shape[dim];
                {{WALKER_STEP_OUTER}}
            }
            // Walker step OUTER / operand offset - end

            {{PRAGMA_SIMD}}
            for (int64_t iidx=0; iidx < chunksize; iidx++) {
                // Apply operator(s) on operands - begin
                {{OPERATIONS}}
                // Apply operator(s) on operands - end

                // Walker step INNER - begin
                {{WALKER_STEP_INNER}}
                // Walker step INNER - end
            }
			// Write EXPANDED scalars back to memory - begin
            if (0==tid) {
                {{WRITE_EXPANDED_SCALARS}}
            }
			// Write EXPANDED scalars back to memory - end
        }
        // Accumulator SYNC - begin
        {{ACCU_OPD_SYNC}}
        // Accumulator SYNC - end
        }
    }
}

