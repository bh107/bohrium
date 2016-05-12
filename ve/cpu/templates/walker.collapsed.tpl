//
//  walker.collapsed
//
//    Flattens the iteration-space and walks it using a single loop construct.
//    Work is partitioned in the number of elements, regardless of dimension.
//
//
{{OFFLOAD_BLOCK}}
{
    const int mthreads       = omp_get_max_threads();
    const int64_t nworkers   = iterspace_nelem > mthreads ? mthreads : 1;
    const int64_t work_split = iterspace_nelem / nworkers;
    const int64_t work_spill = iterspace_nelem % nworkers;

    // Acculumator INNER_DIM - end
    {{WALKER_INNER_DIM}}
    // Acculumator INNER_DIM - end

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
            // Walker declaration(s) - begin
            {{WALKER_DECLARATION}}
            // Walker declaration(s) - end

            // Stride of innermost dimension - begin
            {{WALKER_STRIDE_INNER}}
            // Stride of innermost dimension - end

            // Walker offset(s) - begin
            {{WALKER_OFFSET}}
            // Walker offset(s) - end

            // Accumulator DECLARE - begin
            {{ACCU_LOCAL_DECLARE_COMPLETE}}
            {{ACCU_LOCAL_DECLARE_PARTIAL}}
            // Accumulator DECLARE - end

            {{PRAGMA_SIMD}}
            for (int64_t eidx = work_offset; eidx<work_end; ++eidx) {
                // Apply operator(s) on operands - begin
                {{OPERATIONS}}
                // Apply operator(s) on operands - end

                // Walker step INNER - begin
                {{WALKER_STEP_INNER}}
                // Walker step INNER - end
            }
            
            // Accumulator SYNC - begin
            {{ACCU_OPD_SYNC_COMPLETE}}
            {{ACCU_OPD_SYNC_PARTIAL}}
            // Accumulator SYNC - end

            if (0==tid) {
                {{WRITE_EXPANDED_SCALARS}}
            }
        }
    }
    // Accumulator COMPLETE WRITEBACK - begin
    {{ACCU_OPD_WRITEBACK}}
    // Accumulator COMPLETE WRITEBACK - end
}
