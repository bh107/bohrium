//
//  Collapsed walker (OpenACC)
//
{{OFFLOAD}}
{
    // Walker declaration(s) - begin
    {{WALKER_DECLARATION}}
    // Walker declaration(s) - end

    int64_t shape = iterspace_shape[0];

    {{OFFLOAD_LOOP}}
    for (int64_t eidx = 0; eidx<shape; ++eidx) {
        // Apply operator(s) on operands - begin
        {{OPERATIONS}}
        // Apply operator(s) on operands - end
    }
}
