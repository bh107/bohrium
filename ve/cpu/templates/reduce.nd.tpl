// Reduction operation of a strided n-dimensional array where n>1
// TODO: openmp
//       dimension-based optimizations
//       loop collapsing...
{
    int64_t nelements = 1;
    for(int k=0; k<a{{NR_OUTPUT}}_ndim; ++k) {
        nelements *= a{{NR_OUTPUT}}_shape[k];
    }

    const int64_t last_dim  = a{{NR_OUTPUT}}_ndim-1;
    const int64_t shape_ld  = a{{NR_OUTPUT}}_shape[last_dim];
    const int64_t last_e    = nelements-1;

    {{#OPERAND}}
    {{#SCALAR}}{{TYPE}} a{{NR}}_current = *a{{NR}}_first;{{/SCALAR}}
    {{#SCALAR_CONST}}const {{TYPE}} a{{NR}}_current = *a{{NR}}_first;{{/SCALAR_CONST}}
    {{#SCALAR_TEMP}}{{TYPE}} a{{NR}}_current;{{/SCALAR_TEMP}}
    {{#ARRAY}}int64_t  a{{NR}}_stride_ld  = a{{NR}}_stride[last_dim];{{/ARRAY}}
    {{/OPERAND}}

    const int mthreads = omp_get_max_threads();
    const int64_t nworkers = nelements > mthreads ? mthreads : 1;

    int64_t coord[CPU_MAXDIM];
    memset(coord, 0, CPU_MAXDIM * sizeof(int64_t));

    {{TYPE_AXIS}} axis = *a{{NR_SINPUT}}_first;

    // Create a stride-structure for the input without the axis dimension
    int64_t stride[CPU_MAXDIM];
    for(int i=0,t=0; i<a{{NR_FINPUT}}_ndim; ++i) {
        if (i==axis) {  // Skip the axis dimension
            continue;
        }
        stride[t] = a{{NR_FINPUT}}_stride[i];
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
        {{TYPE_OUTPUT}}* a{{NR_OUTPUT}}_current = a{{NR_OUTPUT}}_first;
        for (int64_t j=0; j<=last_dim; ++j) {           
            a{{NR_OUTPUT}}_current += coord[j] * a{{NR_OUTPUT}}_stride[j];
        }

        //
        // Iterate over "last" / "innermost" dimension
        //
        for (int64_t j = 0; j < shape_ld; ++j) {

            //
            // Walk over the input
            {{TYPE_INPUT}} *a{{NR_FINPUT}}_current = a{{NR_FINPUT}}_first;
            // Increment the input-offset based on every but the axis dimension
            for(int64_t s=0; s<a{{NR_FINPUT}}_ndim-1; ++s) {
                a{{NR_FINPUT}}_current += coord[s] * stride[s];
            }
            a{{NR_FINPUT}}_current += j*stride[last_dim];
            // Non-axis offset ready
            //
            
            //
            // Do the reduction over the axis dimension
            //
            {{TYPE_OUTPUT}} state = *a{{NR_FINPUT}}_current;
            for(int64_t k=1; k<a{{NR_FINPUT}}_shape[axis]; ++k) {
                //
                // Walk to the next element input-element along the axis dimension
                a{{NR_FINPUT}}_current += a{{NR_FINPUT}}_stride[axis];

                //
                // Apply the operator
                //
                state += *a{{NR_FINPUT}}_current;
            }
            // Write the accumulation output
            *a{{NR_OUTPUT}}_current = state;

            // Now increment the output
            a{{NR_OUTPUT}}_current += a{{NR_OUTPUT}}_stride_ld;
        }
        cur_e += shape_ld;

        // 
        // coord[last_dim] is never used, only all the other coord[dim!=last_dim]
        for (int64_t j = last_dim-1; j >= 0; --j) {         // Increment coordinates for the remaining dimensions
            coord[j]++;
            if (coord[j] < a{{NR_OUTPUT}}_shape[j]) {       // Still within this dimension
                break;
            } else {            // Reached the end of this dimension
                coord[j] = 0;   // Reset coordinate
            }                   // Loop then continues to increment the next dimension
        }
    }
}

