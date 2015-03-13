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

    int64_t weight[CPU_MAXDIM]; // Helper for step-calculation
    int64_t acc = 1;
    weight[axis] = acc;
    for(int64_t idx=0; idx<iterspace->ndim; ++idx) {
        if(idx==axis) {
            continue;
        }
        acc *= iterspace->shape[idx];
        weight[idx] = acc;
    }

    const int mthreads          = omp_get_max_threads();
    const int64_t chunksize     = iterspace->shape[axis];
    const int64_t nchunks       = iterspace->nelem / chunksize;
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

        /*
        printf(
            "tid=%d, axis=%ld, nchunks=%ld, chunksize=%ld, work_offset=%ld, work_end=%ld\n",
            tid, axis, nchunks, chunksize, work_offset, work_end
        );*/

        if (work) {
        // Walker STRIDE_INNER - begin
        const uint64_t axis_stride = {{OPD_IN1}}_stride[axis];
        // Walker STRIDE_INNER - end

        const int64_t eidx_begin = work_offset*chunksize;
        const int64_t eidx_end   = work_end*chunksize;
        int64_t coord[CPU_MAXDIM] = {0};
        for(int64_t eidx=0; eidx<eidx_begin; eidx+=chunksize) {
            for(int64_t dim=iterspace->ndim-1; dim>=0; --dim) {
                if (dim==axis) {
                    continue;
                }
                coord[dim] = (coord[dim]+1)% iterspace->shape[dim];
                //printf("coord[%ld]=%ld\n", dim, coord[dim]);
            }
        }

        for(int64_t eidx=eidx_begin; eidx<eidx_end; eidx+=chunksize) {
            // Walker declaration(s) - begin
            {{ETYPE}}* {{OPD_IN1}} = {{OPD_IN1}}_first;
            {{ETYPE}}* {{OPD_OUT}} = {{OPD_OUT}}_first;
            // Walker declaration(s) - end

            // Walker step non-axis / operand offset - begin
            int64_t froyo = 0;
            for(int64_t dim=iterspace->ndim-1; dim>=0; --dim) {
                if (dim==axis) {
                    continue;
                }
                //const int64_t coord = (eidx / weight[dim]) % iterspace->shape[dim];
                /*
                printf(
                    "eidx=%ld, dim=%ld, coord = %ld, weight=%ld, shape=%ld\n", 
                    eidx, dim, coord, weight[dim], iterspace->shape[dim]
                );*/
                {{OPD_IN1}} += coord[dim] * {{OPD_IN1}}_stride[dim];
                {{OPD_OUT}} += coord[dim] * {{OPD_OUT}}_stride[froyo];
                froyo++;

                coord[dim] = (coord[dim]+1)% iterspace->shape[dim];
            }
            // Walker step non-axis / operand offset - end

            {{ETYPE}} accu = {{NEUTRAL_ELEMENT}};
            {{PRAGMA_SIMD}}
            for (int64_t aidx=0; aidx < chunksize; aidx++) {
                // Apply operator(s) on operands - begin
                {{REDUCE_OPER}}
                // Apply operator(s) on operands - end

                // Walker step INNER - begin
                {{OPD_IN1}} += axis_stride;
                // Walker step INNER - end
            }
            *{{OPD_OUT}} = accu;
            //*{{OPD_OUT}} = 1;
            printf("jazz = %f\n", accu);
        }}
    }
}

