//
// Scan operation on one-dimensional arrays with strided access.
//
{{OFFLOAD_BLOCK}}
{
    {{WALKER_INNER_DIM}}
    {{WALKER_DECLARATION}}
    // Walker STRIDE_INNER - begin
    {{WALKER_STRIDE_INNER}}
    // Walker STRIDE_INNER - end

    {{ACCU_LOCAL_DECLARE}}
    for(int64_t j=0; j<iterspace_shape[0]; ++j) {
        {{OPERATIONS}}
       
        // Walker step INNER - begin
        {{WALKER_STEP_INNER}}
        // Walker step INNER - end
    }

    // TODO: Handle write-out of non-temp and non-const scalars.
}

