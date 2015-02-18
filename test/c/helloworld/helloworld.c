#include <bh_c.h>
#include <stdio.h>

void compute()
{
    bh_multi_array_float32_p output;
    bh_multi_array_float32_p a;
    bh_multi_array_float32_p b;
    bh_multi_array_uint32_p r_flat;
    bh_multi_array_uint32_p r_shaped;
    float* data;

    int64_t i, j;

    int64_t shape[2] = {3,3};
    int64_t stride[2] = {3,1};

    // Sequence of ones
    a = bh_multi_array_float32_new_ones(2, shape);

    // Range from [0 - 9[
    r_flat = bh_multi_array_uint32_new_range(10);

    // Reshaped to 3x3
    r_shaped = bh_multi_array_uint32_new_view(r_flat, 2, 0, shape, stride);

    // Make into floats
    b = bh_multi_array_float32_new_empty(2, shape);
    bh_multi_array_float32_identity_uint32(b, r_shaped);

    // Do actual computation
    output = bh_multi_array_float32_new_empty(2, shape);
    bh_multi_array_float32_add(output, a, b);

    // Issue a sync instruction to ensure data is present in local memory space
    bh_multi_array_float32_sync(output);

    // Execute all pending instructions, including the sync command
    bh_runtime_flush();

    // Grab the result data
    data = bh_multi_array_float32_get_data(output);

    // Print out the result
    printf("Adding ones to range in 2D: \n");
    for(i = 0; i < shape[0]; i++) {
        for(j = 0; j < shape[1]; j++)
            printf("%f, ", data[(i*stride[0])+(j*stride[1])]);

        printf("\n");
    }

    // Clean up anything that was allocated
    bh_multi_array_float32_destroy(a);
    bh_multi_array_float32_destroy(b);
    bh_multi_array_float32_destroy(output);
    bh_multi_array_uint32_destroy(r_flat);
    bh_multi_array_uint32_destroy(r_shaped);
}

int main()
{
    compute();
    bh_runtime_shutdown();
    return 0;
}
