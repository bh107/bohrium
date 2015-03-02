// Scan operation of a strided n-dimensional array where n>1
// TODO: openmp
//       dimension-based optimizations
//       loop collapsing...
{
    const int64_t nelements = iterspace->nelem;
    {{TYPE_AXIS}} axis = *a{{NR_SINPUT}}_first;

    const int64_t last_e      = nelements-1;
    const int64_t shape_axis  = iterspace->shape[axis];
    const int64_t ndim        = iterspace->ndim;

    {{#OPERAND}}
    {{#SCALAR}}{{TYPE}} a{{NR}}_current = *a{{NR}}_first;{{/SCALAR}}
    {{#SCALAR_CONST}}const {{TYPE}} a{{NR}}_current = *a{{NR}}_first;{{/SCALAR_CONST}}
    {{#SCALAR_TEMP}}{{TYPE}} a{{NR}}_current;{{/SCALAR_TEMP}}
    {{#ARRAY}}int64_t a{{NR}}_stride_axis = a{{NR_OUTPUT}}_stride[axis];{{/ARRAY}}
    {{/OPERAND}}

    const int mthreads = omp_get_max_threads();
    const int64_t nworkers = nelements > mthreads ? mthreads : 1;

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
            if (coord[j] < iterspace->shape[j]) {       
                break;
            } else {            // Reached the end of this dimension
                coord[j] = 0;   // Reset coordinate
            }                   // Loop then continues to increment the next dimension
        }
    }
    // TODO: Handle write-out of non-temp and non-const scalars.
}

