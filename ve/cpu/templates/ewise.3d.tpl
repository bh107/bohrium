//
// Codegen template is used for:
//
//	* MAP|ZIP|GENERATE|FLOOD on 3D strided arrays.
//
{
    const int64_t last_dim = iterspace->ndim-1;

    int64_t weight[CPU_MAXDIM]; // Helper for step-calculation
    int acc = 1;
    for(int idx=last_dim; idx >=0; --idx) {
        weight[idx] = acc;
        acc *= iterspace->shape[idx];
    }

    const int mthreads = omp_get_max_threads();
    const int64_t nworkers = iterspace->shape[0] > mthreads ? mthreads : 1;
    const int64_t work_split= iterspace->shape[0] / nworkers;
    const int64_t work_spill= iterspace->shape[0] % nworkers;

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

        for (int64_t tld_idx=work_offset; tld_idx<work_end; ++tld_idx) {
            for (int64_t sld_idx = 0; sld_idx<iterspace->shape[1]; ++sld_idx) {
                for (int64_t ld_idx = 0; ld_idx<iterspace->shape[2]; ++ld_idx) {
	                // Apply operator(s)
                    {{OPERATIONS}}

                    // Increment operands
                    {{WALKER_STEP_LD}}
                }
                // Increment operands
                {{WALKER_STEP_SLD}}
            }
            // Increment operands
            {{WALKER_STEP_TLD}}
        }}
    }
    // TODO: Handle write-out of non-temp and non-const scalars.
}

