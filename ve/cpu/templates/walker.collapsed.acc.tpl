//
//  Collapsed walker (OpenACC)
//
{{OFFLOAD_BLOCK}}
{
    // Walker declaration(s) - begin
    {{WALKER_DECLARATION}}
    // Walker declaration(s) - end

    // Accumulator DECLARE - begin
    {{ACCU_LOCAL_DECLARE_COMPLETE}}
    {{ACCU_LOCAL_DECLARE_PARTIAL}}
    // Accumulator DECLARE - end

    int64_t shape = iterspace_shape[0];

    {{OFFLOAD_LOOP}}
    for (int64_t eidx = 0; eidx<shape; ++eidx) {
        // Apply operator(s) on operands - begin
        {{OPERATIONS}}
        // Apply operator(s) on operands - end
    }
    {{OFFLOAD_LOOP_SEQUEL}}
}
