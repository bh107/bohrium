//
// Elementwise operation on two-dimensional arrays using strided indexing
{
    const int64_t shape_ld  = iterspace->shape[1];
    const int64_t shape_sld = iterspace->shape[0];

    const int mthreads = omp_get_max_threads();
    const int64_t nworkers = shape_sld > mthreads ? mthreads : 1;
    const int64_t work_split= shape_sld / nworkers;
    const int64_t work_spill= shape_sld % nworkers;

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
		if (work) {
        // Operand declaration(s)
        {{WALKER_DECLARATION}}
        // Operand offsets(s)
        {{WALKER_OFFSET}}
        // Stepsize
        {{WALKER_STEPSIZE}}

        for(int64_t j=work_offset; j<work_end; ++j) {
            for (int64_t i = 0; i < shape_ld; ++i) {
                // Apply operator(s)
                {{OPERATIONS}}
                
                // Increment operands
                {{WALKER_STEP_LD}}
            }
            // Increment operands
            {{WALKER_STEP_SLD}}

        }}
    }
    // TODO: Handle write-out of non-temp and non-const scalars.
}

