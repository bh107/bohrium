// Scan operation of a strided n-dimensional array where n>1
{
    const int64_t nelements = iterspace_nelem;
    {{ATYPE}} axis = {{OPD_IN2}}_data;

    const int64_t last_e      = nelements-1;
    const int64_t shape_axis  = iterspace_shape[axis];
    const int64_t ndim        = iterspace_ndim;

    int64_t coord[CPU_MAXDIM];
    memset(coord, 0, CPU_MAXDIM * sizeof(int64_t));

    int64_t {{OPD_IN1}}_stride_axis = {{OPD_OUT}}_strides[axis];
    int64_t {{OPD_OUT}}_stride_axis = {{OPD_OUT}}_strides[axis];

    //
    //  Walk over the output
    //
    int64_t cur_e = 0;
    while (cur_e <= last_e) {

        //
        // Compute offset based on coordinate
        //
        {{ETYPE}}* {{OPD_OUT}} = {{OPD_OUT}}_data + {{OPD_OUT}}_start;
        {{ETYPE}}* {{OPD_IN1}} = {{OPD_IN1}}_data + {{OPD_IN1}}_start;

        for (int64_t j=0; j<ndim; ++j) {           
            {{OPD_OUT}} += coord[j] * {{OPD_OUT}}_strides[j];
            {{OPD_IN1}} += coord[j] * {{OPD_IN1}}_strides[j];
        }

        //
        // Iterate over axis dimension
        //
        {{ACCU_LOCAL_DECLARE}}
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
            if (coord[j] < iterspace_shape[j]) {       
                break;
            } else {            // Reached the end of this dimension
                coord[j] = 0;   // Reset coordinate
            }                   // Loop then continues to increment the next dimension
        }
    }
    // TODO: Handle write-out of non-temp and non-const scalars.
}

