//
//  walking.axis.nd
//
//	Walks the iteration-space using other/axis loop constructs.
//	Partitions work into chunks of size equal to the "axis" dimension.
//
{{OFFLOAD_BLOCK}}
{
    {{WALKER_AXIS_DIM}}
    int64_t axis_shape = iterspace_shape[axis_dim];

    //
    // Construct an iteration-space that does not include the axis-dimension
    int64_t shape[CPU_MAXDIM] = {0};
    for(int64_t dim=0, outer_dim = 0; dim < iterspace_ndim; ++dim) {
        if (dim == axis_dim) {
            continue;
        }
        shape[outer_dim] = iterspace_shape[dim];
        ++outer_dim;
    }

    int64_t ndim      = iterspace_ndim-1;
    int64_t inner_dim = ndim-1;
    int64_t nelements = 1;
    for(int64_t dim=0; dim<ndim; ++dim) {
        nelements *= shape[dim];
    }

    // Now only the strides a needed
    //
    // Compute the weight
    int64_t weight[CPU_MAXDIM];
    int64_t acc = 1;
    for(int64_t dim=inner_dim; dim >=0; --dim) {
        weight[dim] = acc;
        acc *= shape[dim];
    }

    const int mthreads       = omp_get_max_threads();
    const int64_t chunksize  = shape[inner_dim];
    const int64_t nchunks    = nelements / chunksize;
    const int64_t nworkers   = nchunks > mthreads ? mthreads : 1;
    const int64_t work_split = nchunks / nworkers;
    const int64_t work_spill = nchunks % nworkers;

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
            // Walker STRIDE_AXIS - begin
            {{WALKER_STRIDE_AXIS}}
            // Walker STRIDE_AXIS - end

            // Accumulator DECLARE COMPLETE - begin
            {{ACCU_LOCAL_DECLARE_COMPLETE}}
            // Accumulator DECLARE COMPLETE - end

            const int64_t eidx_begin = work_offset*chunksize;
            const int64_t eidx_end   = work_end*chunksize;
            for(int64_t eidx=eidx_begin; eidx<eidx_end; ++eidx) {
                // Walker declaration(s) - begin
                {{WALKER_DECLARATION}}
                // Walker declaration(s) - end

                // Walker step non-axis / operand offset - begin
                for(int64_t dim=0, other_dim=0; dim<iterspace_ndim; ++dim) {
                    if (dim==axis_dim) {
                        continue;
                    }
                    const int64_t coord = (eidx / weight[other_dim]) % shape[other_dim];

                    {{WALKER_STEP_OTHER}}
                    ++other_dim;
                }
                // Walker step non-axis / operand offset - end

                // Accumulator DECLARE PARTIAL - begin
                {{ACCU_LOCAL_DECLARE_PARTIAL}}
                // Accumulator DECLARE PARTIAL - end

                {{PRAGMA_SIMD}}
                for (int64_t aidx=0; aidx < axis_shape; aidx++) {
                    // Apply operator(s) on operands - begin
                    {{OPERATIONS}}
                    // Apply operator(s) on operands - end

                    // Walker step INNER - begin
                    {{WALKER_STEP_AXIS}}
                    // Walker step INNER - end
                }

                // Accumulator PARTIAL SYNC - begin
                {{ACCU_OPD_SYNC_PARTIAL}}
                // Accumulator PARTIAL SYNC - end
            }

            // Write EXPANDED scalars back to memory - begin
            if (0==tid) {
                {{WRITE_EXPANDED_SCALARS}}
            }
            // Write EXPANDED scalars back to memory - end

            // Accumulator COMPLETE SYNC - begin
            {{ACCU_OPD_SYNC_COMPLETE}}
            // Accumulator COMPLETE SYNC - end
        }
    }
}
