//
//  Collapsed walker (OpenACC)
//
/*
{{OFFLOAD}}
*/
{
    // Walker declaration(s) - begin
    {{WALKER_DECLARATION}}
    // Walker declaration(s) - end

    #pragma acc kernels loop
    for (int64_t eidx = 0; eidx<iterspace_shape[0]; ++eidx) {
        // Apply operator(s) on operands - begin
        {{OPERATIONS}}
        // Apply operator(s) on operands - end
    }
}
