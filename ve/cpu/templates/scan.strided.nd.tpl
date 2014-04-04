// Scan operation of a strided n-dimensional array where n>1
// TODO: openmp
//       dimension-based optimizations
//       loop collapsing...
{
    int64_t nelements = 1;
    for(int k=0; k<a{{NR_OUTPUT}}_ndim; ++k) {
        nelements *= a{{NR_OUTPUT}}_shape[k];
    }
    {{TYPE_AXIS}} axis = *a{{NR_SINPUT}}_first;

    int64_t last_e      = nelements-1;
    int64_t shape_axis  = a{{NR_OUTPUT}}_shape[axis];
    int64_t ndim        = a{{NR_OUTPUT}}_ndim;

    {{#OPERAND}}{{#ARRAY}}
    int64_t  a{{NR}}_stride_axis = a{{NR_OUTPUT}}_stride[axis];
    {{/ARRAY}}{{/OPERAND}}

    int mthreads = omp_get_max_threads();
    int64_t nworkers = nelements > mthreads ? mthreads : 1;

    int64_t coord[CPU_MAXDIM];
    memset(coord, 0, CPU_MAXDIM * sizeof(int64_t));

    //
    //  Walk over the output
    //
    int64_t cur_e = 0;
    while (cur_e <= last_e) {

        //
        // Compute offset based on coordinate
        //
        {{TYPE_OUTPUT}}* a{{NR_OUTPUT}}_current = a{{NR_OUTPUT}}_first;
        {{TYPE_INPUT}}*  a{{NR_FINPUT}}_current = a{{NR_FINPUT}}_first;

        for (int64_t j=0; j<ndim; ++j) {           
            a{{NR_OUTPUT}}_current += coord[j] * a{{NR_OUTPUT}}_stride[j];
            a{{NR_FINPUT}}_current += coord[j] * a{{NR_FINPUT}}_stride[j];
        }

        //
        // Iterate over axis dimension
        //
        {{TYPE_INPUT}} state = ({{TYPE_INPUT}}){{NEUTRAL_ELEMENT}};
        for (int64_t j = 0; j<shape_axis; ++j) {
            {{#OPERATORS}}
            {{OPERATOR}};
            {{/OPERATORS}}

            // Update the output
            *a{{NR_OUTPUT}}_current = state;
            
            a{{NR_OUTPUT}}_current += a{{NR_OUTPUT}}_stride_axis;
            a{{NR_FINPUT}}_current += a{{NR_FINPUT}}_stride_axis;
        }
        cur_e += shape_axis;

        // Increment coordinates for the remaining dimensions
        for (int64_t j = ndim-1; j >= 0; --j) {
            if (j==axis) {      // Skip the axis dimension
                continue;       // It is calculated within the loop above
            }
            coord[j]++;         // Still within this dimension
            if (coord[j] < a{{NR_OUTPUT}}_shape[j]) {       
                break;
            } else {            // Reached the end of this dimension
                coord[j] = 0;   // Reset coordinate
            }                   // Loop then continues to increment the next dimension
        }
    }
    
    {{#OPERAND}}{{#SCALAR}}
    // Write scalar-operand to main-memory;
    // Note this is only necessary for non-temporary scalar-operands.
    // So this code should only be generated for non-temps.
    if ({{NR_OUTPUT}} == {{NR}}) {
        *a{{NR}}_first = a{{NR}}_current;
    }
    {{/SCALAR}}{{/OPERAND}}
}

