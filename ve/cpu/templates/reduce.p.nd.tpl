//
// Codegen template is used for:
//
//	* REDUCE_PARTIAL on arrays of any LAYOUT and rank > 1.
//
//	Partitions work into 1D reductions over the axis.
//	Distribites work staticly/evenly among threads.
//
{
    {{ATYPE}} axis = *{{OPD_IN2}}_first;

    int64_t shape_axis = iterspace->shape[axis];
    //
    // Construct an iteration-space that does not include the axis-dimension
    int64_t shape[CPU_MAXDIM] = {0};
    for(int64_t dim=0, outer_dim = 0; dim < iterspace->ndim; ++dim) {
        if (dim == axis) {
            continue;
        }
        shape[outer_dim] = iterspace->shape[dim];
        ++outer_dim;
    }
    int64_t ndim = iterspace->ndim-1;
    int64_t last_dim = ndim-1;
    int64_t nelements = 1;
    for(int64_t dim=0; dim<ndim; ++dim) {
        nelements *= shape[dim];
    }
    // Now only the strides a needed
    //
    // Compute the weight
    int64_t weight[CPU_MAXDIM];
    int64_t acc = 1;
    for(int64_t dim=last_dim; dim >=0; --dim) {
        weight[dim] = acc;
        acc *= shape[dim];
    }

    const int mthreads          = omp_get_max_threads();
    const int64_t chunksize     = shape[last_dim];
    const int64_t nchunks       = nelements / chunksize;
    const int64_t nworkers      = nchunks > mthreads ? mthreads : 1;
    const int64_t work_split    = nchunks / nworkers;
    const int64_t work_spill    = nchunks % nworkers;
    
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
        const uint64_t axis_stride = {{OPD_IN1}}_stride[axis];
        // Walker STRIDE_INNER - end

        const int64_t eidx_begin = work_offset*chunksize;
        const int64_t eidx_end   = work_end*chunksize;
        for(int64_t eidx=eidx_begin; eidx<eidx_end; ++eidx) {
            // Walker declaration(s) - begin
            {{ETYPE}}* {{OPD_IN1}} = {{OPD_IN1}}_first;
            {{ETYPE}}* {{OPD_OUT}} = {{OPD_OUT}}_first;
            // Walker declaration(s) - end

            // Walker step non-axis / operand offset - begin
            for(int64_t dim=0, other_dim=0; dim<iterspace->ndim; ++dim) {
                if (dim==axis) {
                    continue;
                }
                const int64_t coord = (eidx / weight[other_dim]) % shape[other_dim];

                {{OPD_IN1}} += coord * {{OPD_IN1}}_stride[dim];
                {{OPD_OUT}} += coord * {{OPD_OUT}}_stride[other_dim];
                ++other_dim;
            }
            // Walker step non-axis / operand offset - end

            {{ETYPE}} accu = {{NEUTRAL_ELEMENT}};
            {{PRAGMA_SIMD}}
            for (int64_t aidx=0; aidx < shape_axis; aidx++) {
                // Apply operator(s) on operands - begin
                {{OPERATIONS}}
                // Apply operator(s) on operands - end

                // Walker step INNER - begin
                {{OPD_IN1}} += axis_stride;
                // Walker step INNER - end
            }
            *{{OPD_OUT}} = accu;
        }}
    }
}

