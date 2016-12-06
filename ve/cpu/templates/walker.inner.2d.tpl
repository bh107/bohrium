//
//  walker.inner.2d
//
//    Walks the iteration-space using outer/inner loop constructs.
//    Partitions work into chunks of size equal to the inner-most dimension.
//
{{OFFLOAD_BLOCK}}
{
    const int64_t inner_dim  = iterspace_ndim-1;
    const int64_t outer_dim  = iterspace_ndim-2;

    const int mthreads       = omp_get_max_threads();
    const int64_t chunksize  = iterspace_shape[inner_dim];
    const int64_t nchunks    = iterspace_nelem / chunksize;
    const int64_t nworkers   = nchunks > mthreads ? mthreads : 1;
    const int64_t work_split = nchunks / nworkers;
    const int64_t work_spill = nchunks % nworkers;

    // Acculumator INIT - begin
    {{ACCU_OPD_INIT}}
    // Acculumator INIT - end

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
            // Accumulator DECLARE COMPLETE - begin
            {{ACCU_LOCAL_DECLARE_COMPLETE}}
            // Accumulator DECLARE COMPLETE - end

            // Walker declaration(s) - begin
            {{WALKER_DECLARATION}}
            // Walker declaration(s) - end

            // Walker STRIDE_OUTER - begin
            {{WALKER_STRIDE_OUTER}}
            // Walker STRIDE_OUTER - end

            // Walker STRIDE_OUTER - begin
            {{WALKER_STRIDE_INNER}}
            // Walker STRIDE_OUTER - end

            // Walker offset(s) - begin
            {{WALKER_OFFSET}}
            // Walker offset(s) - end

            // Iteration space
            const int64_t eidx_begin = work_offset*chunksize;
            const int64_t eidx_end   = work_end*chunksize;
            for(int64_t eidx=eidx_begin; eidx<eidx_end; eidx+=chunksize) {

                // Accumulator DECLARE PARTIAL - begin
                {{ACCU_LOCAL_DECLARE_PARTIAL}}
                // Accumulator DECLARE PARTIAL - end

                {{PRAGMA_SIMD}}
                for (int64_t iidx=0; iidx < chunksize; iidx++) {
                    // Apply operator(s) on operands - begin
                    {{OPERATIONS}}
                    // Apply operator(s) on operands - end

                    // Walker step INNER - begin
                    {{WALKER_STEP_INNER}}
                    // Walker step INNER - end
                }
                // Accumulator PARTIAL SYNC - begin
                {{ACCU_OPD_SYNC_PARTIAL}}
                // Accumulator PARTIAL SYNC - end

                // Walker step OUTER / operand offset - begin
                {{WALKER_STEP_OUTER}}
                // Walker step OUTER / operand offset - end

                // Write EXPANDED scalars back to memory - begin
                if (0==tid) {
                    {{WRITE_EXPANDED_SCALARS}}
                }
                // Write EXPANDED scalars back to memory - end
            }
            // Accumulator COMPLETE SYNC - begin
            {{ACCU_OPD_SYNC_COMPLETE}}
            // Accumulator COMPLETE SYNC - end
        }
    }
    // Accumulator COMPLETE WRITEBACK - begin
    {{ACCU_OPD_WRITEBACK}}
    // Accumulator COMPLETE WRITEBACK - end
}
