// Reduction operation of a strided n-dimensional array where n>1
// TODO: openmp
//       dimension-based optimizations
//       loop collapsing...
{
    int64_t nelements = 1;
    for(int k=0; k<{{OPD_OUT}}_ndim; ++k) {
        nelements *= {{OPD_OUT}}_shape[k];
    }

    const int64_t last_dim  = {{OPD_OUT}}_ndim-1;
    const int64_t shape_ld  = {{OPD_OUT}}_shape[last_dim];
    const int64_t last_e    = nelements-1;

    // Operand declaration(s)
    {{WALKER_DECLARATION}}
    {{WALKER_STRIDES}}

    const int mthreads = omp_get_max_threads();
    const int64_t nworkers = nelements > mthreads ? mthreads : 1;

    int64_t coord[CPU_MAXDIM];
    memset(coord, 0, CPU_MAXDIM * sizeof(int64_t));

    {{ATYPE}} axis = *{{OPD_IN2}}_first;

    // Create a stride-structure for the input without the axis dimension
    int64_t stride[CPU_MAXDIM];
    for(int i=0,t=0; i<{{OPD_IN1}}_ndim; ++i) {
        if (i==axis) {  // Skip the axis dimension
            continue;
        }
        stride[t] = {{OPD_IN1}}_stride[i];
        t++;
    }

    //
    //  Walk over the output
    //
    int64_t cur_e = 0;
    while (cur_e <= last_e) {

        //
        // Compute offset based on coordinate
        //
        {{ETYPE}}* {{OPD_OUT}}_current = {{OPD_OUT}}_first;
        for (int64_t j=0; j<=last_dim; ++j) {           
            {{OPD_OUT}}_current += coord[j] * {{OPD_OUT}}_stride[j];
        }

        //
        // Iterate over "last" / "innermost" dimension
        //
        for (int64_t j=0; j<shape_ld; ++j) {

            //
            // Walk over the input
            {{TYPE_INPUT}} *{{OPD_IN1}}_current = {{OPD_IN1}}_first;
            // Increment the input-offset based on every but the axis dimension
            for(int64_t s=0; s<{{OPD_IN1}}_ndim-1; ++s) {
                {{OPD_IN1}}_current += coord[s] * stride[s];
            }
            {{OPD_IN1}}_current += j*stride[last_dim];
            // Non-axis offset ready
            //
            
            //
            // Do the reduction over the axis dimension
            //
            {{ETYPE}} state = *{{OPD_IN1}}_current;
            for(int64_t k=1; k<{{OPD_IN1}}_shape[axis]; ++k) {
                //
                // Walk to the next element input-element along the axis dimension
                {{OPD_IN1}}_current += {{OPD_IN1}}_stride[axis];

                //
                // Apply the operator
                //
                state += *{{OPD_IN1}}_current;
            }
            // Write the accumulation output
            *{{OPD_OUT}}_current = state;

            // Now increment the output
            {{OPD_OUT}}_current += {{OPD_OUT}}_stride_ld;
        }
        cur_e += shape_ld;

        // 
        // coord[last_dim] is never used, only all the other coord[dim!=last_dim]
        for (int64_t j = last_dim-1; j >= 0; --j) {         // Increment coordinates for the remaining dimensions
            coord[j]++;
            if (coord[j] < {{OPD_OUT}}_shape[j]) {       // Still within this dimension
                break;
            } else {            // Reached the end of this dimension
                coord[j] = 0;   // Reset coordinate
            }                   // Loop then continues to increment the next dimension
        }
    }
}

