// Scan operation of a strided n-dimensional array where n>1
// TODO: openmp
//       dimension-based optimizations
//       loop collapsing...
{
    const int64_t nelements = iterspace->nelem;
    {{ATYPE}} axis = *{{OPD_IN2}}_first;

    const int64_t last_e      = nelements-1;
    const int64_t shape_axis  = iterspace->shape[axis];
    const int64_t ndim        = iterspace->ndim;

    int64_t coord[CPU_MAXDIM];
    memset(coord, 0, CPU_MAXDIM * sizeof(int64_t));

    int64_t* const {{OPD_IN1}}_stride = args[2]->stride;
    int64_t* const {{OPD_OUT}}_stride = args[1]->stride;
    int64_t {{OPD_IN1}}_stride_axis = {{OPD_OUT}}_stride[axis];
    int64_t {{OPD_OUT}}_stride_axis = {{OPD_OUT}}_stride[axis];

    //
    //  Walk over the output
    //
    int64_t cur_e = 0;
    while (cur_e <= last_e) {

        //
        // Compute offset based on coordinate
        //
        {{ETYPE}}* {{OPD_OUT}} = {{OPD_OUT}}_first;
        {{ETYPE}}* {{OPD_IN1}} = {{OPD_IN1}}_first;

        for (int64_t j=0; j<ndim; ++j) {           
            {{OPD_OUT}} += coord[j] * {{OPD_OUT}}_stride[j];
            {{OPD_IN1}} += coord[j] * {{OPD_IN1}}_stride[j];
        }

        //
        // Iterate over axis dimension
        //
        {{ETYPE}} accu = ({{ETYPE}}){{NEUTRAL_ELEMENT}};
        for (int64_t j = 0; j<shape_axis; ++j) {
            {{OPERATIONS}}
            
            {{OPD_OUT}} += {{OPD_OUT}}_stride_axis;
            {{OPD_IN1}} += {{OPD_IN1}}_stride_axis;
        }
        cur_e += shape_axis;

        // Increment coordinates for the remaining dimensions
        for (int64_t j = ndim-1; j >= 0; --j) {
            if (j==axis) {      // Skip the axis dimension
                continue;       // It is calculated within the loop above
            }
            coord[j]++;         // Still within this dimension
            if (coord[j] < iterspace->shape[j]) {       
                break;
            } else {            // Reached the end of this dimension
                coord[j] = 0;   // Reset coordinate
            }                   // Loop then continues to increment the next dimension
        }
    }
    // TODO: Handle write-out of non-temp and non-const scalars.
}

