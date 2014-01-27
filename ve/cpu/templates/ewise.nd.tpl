{
    va_list list;               // Unpack arguments
    va_start(list, tool);

    int64_t *shape      = va_arg(list, int64_t*);
    int64_t ndim        = va_arg(list, int64_t);

    {{#OPERAND}}
    {{TYPE}} *a{{NR}}_current   = va_arg(list, {{TYPE}}*);
    {{#ARRAY}}
    int64_t  a{{NR}}_start   = va_arg(list, int64_t);
    int64_t *a{{NR}}_stride  = va_arg(list, int64_t*);
    {{/ARRAY}}
    {{/OPERAND}}

    va_end(list);

    int64_t nelements = 1;      // Compute number of elements
    int k;
    for (k = 0; k<ndim; ++k){
        nelements *= shape[k];
    }

    int64_t j,                  // Traversal variables
            last_dim    = ndim-1,
            last_e      = nelements-1;

    int64_t cur_e = 0;
    int64_t shape_ld = shape[last_dim];

    {{#OPERAND}}{{#ARRAY}}
    {{TYPE}} *a{{NR}}_first   = a{{NR}}_current;
    int64_t a{{NR}}_stride_ld = a{{NR}}_stride[last_dim];
    {{/ARRAY}}{{/OPERAND}}

    int64_t coord[CPU_MAXDIM];
    memset(coord, 0, CPU_MAXDIM * sizeof(int64_t));

    while (cur_e <= last_e) {
        
        {{#OPERAND}}{{#ARRAY}}
        a{{NR}}_current = a{{NR}}_first + a{{NR}}_start;         // Reset offsets
        {{/ARRAY}}{{/OPERAND}}

        for (j=0; j<=last_dim; ++j) {           // Compute offset based on coordinate
            {{#OPERAND}}{{#ARRAY}}
            a{{NR}}_current += coord[j] * a{{NR}}_stride[j];
            {{/ARRAY}}{{/OPERAND}}
        }

        for (j = 0; j < shape_ld; j++) {        // Iterate over "last" / "innermost" dimension
            {{#LOOP_BODY}}
            {{OPERATOR}};
            {{/LOOP_BODY}}

            {{#OPERAND}}{{#ARRAY}}
            a{{NR}}_current += a{{NR}}_stride_ld;
            {{/ARRAY}}{{/OPERAND}}
        }
        cur_e += shape_ld;

        // coord[last_dim] is never used, only all the other coord[dim!=last_dim]
        for (j = last_dim-1; j >= 0; --j) {  // Increment coordinates for the remaining dimensions
            coord[j]++;
            if (coord[j] < shape[j]) {      // Still within this dimension
                break;
            } else {                        // Reached the end of this dimension
                coord[j] = 0;               // Reset coordinate
            }                               // Loop then continues to increment the next dimension
        }
    }
}

