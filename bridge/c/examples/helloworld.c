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
    r_shaped = bh_multi_array_uint32_new_from_view(bh_multi_array_uint32_get_base(r_flat), 2, 0, shape, stride);

    // Make into floats
    b = bh_multi_array_float32_convert_uint32(r_shaped);

    // Do actual computation
    output = bh_multi_array_float32_add(a, b);

    // Sync and grab data pointer
    bh_multi_array_float32_sync(output);
    data = bh_multi_array_float32_get_base_data(bh_multi_array_float32_get_base(output));

    // Print out the result
    printf("Adding ones to range in 2D: \n");
    for(i = 0; i < shape[0]; i++) {
        for(j = 0; j < shape[1]; j++)
            printf("%f, ", data[(i*stride[0])+(j*stride[1])]);

        printf("\n");
    }

    // And clean up what has not been auto-cleaned
    bh_multi_array_float32_destroy(output);
    bh_multi_array_uint32_destroy(r_flat);
}

int main()
{
    compute();
    return 0;
}
