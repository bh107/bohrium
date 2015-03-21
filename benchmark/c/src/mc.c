#include <inttypes.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <Random123/philox.h>
#include <bp_util.h>

typedef union philox2x32_as_1x64 {
    philox2x32_ctr_t orig;
    uint64_t combined;
} philox2x32_as_1x64_t;

double mcpi(int64_t samples, uint64_t xr_count, uint64_t yr_count, uint64_t key)
{
    uint64_t darts = 0;
    
    #pragma omp parallel for reduction(+:darts)
    for (int64_t eidx=0; eidx<samples; ++eidx) {

        uint64_t x_raw = ((philox2x32_as_1x64_t)philox2x32(
          ((philox2x32_as_1x64_t)( xr_count+eidx )).orig,
          (philox2x32_key_t){ { key } }
        )).combined;
        double x = x_raw;
        x /= 18446744073709551616.000000;

        uint64_t y_raw = ((philox2x32_as_1x64_t)philox2x32(
          ((philox2x32_as_1x64_t)( yr_count+eidx )).orig,
          (philox2x32_key_t){ { key } }
        )).combined;
        double y = y_raw;
        y /= 18446744073709551616.000000;

        darts += (x*x + y*y) <= 1;
    }
    return (double)darts/samples*4;
}

double run_mcpi(int64_t samples, int64_t iterations)
{
    const uint64_t key = 1597416434;    // Philox key
    uint64_t random_count = 0;          // Calls to philox
    uint64_t xr_count = 0;              // x-count offset to philox
    uint64_t yr_count = 0;              // y-count offset to philox

    double pi_accu = 0.0;               // Accumulation over PI-approximations.
    for(int64_t i=0; i<iterations; ++i) {
        xr_count = random_count++;
        yr_count = random_count++;
        pi_accu += mcpi(samples, xr_count, yr_count, key);
    }
    pi_accu /= iterations;              // Approximated value of PI

    return pi_accu;
}

int main(int argc, char** argv)
{
    bp_arguments_type args = parse_args(argc, argv);        // Parse args

    printf(                                                 // Info
        "Running MonteCarlo PI with %d samples for %d trials\n",
        args.sizes[0], args.sizes[1]
    );
    size_t begin = bp_sample_time();
    double res = run_mcpi(args.sizes[0], args.sizes[1]);    // Run
    size_t end = bp_sample_time();
                                                            
    printf("size: %d*%d, elapsed-time: %f\n",               // Report 
        args.sizes[0],
        args.sizes[1],
        (end-begin)/1000000.0
    );
    if (args.verbose) {                                     // Report verbose
        printf("PI = %f\n", res);
    }

    return 0;
}
