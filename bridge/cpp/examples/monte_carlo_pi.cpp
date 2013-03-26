#include <iostream>
#include "bh/bh.hpp"

using namespace std;
using namespace bh;

multi_array<double>& monte_carlo_pi(int samples, int iterations)
{
    multi_array<double> x, y, z, r, sum(1);

    sum = 0.0;                              // Acculumate across iterations
    for(int i=0; i<iterations; ++i) {
        x = random<double>(samples);        // Sample random numbers
        y = random<double>(samples);

        z = sqrt(x * x + y * y);    // Approximate
        r = z <= 1.0;               // Filter

        sum += (r.reduce(ADD,0)*4.0) / (double)samples; // 
    }
    return sum / (double)iterations;
}

int main()
{
    cout << "Pi Approximation: " << monte_carlo_pi(1000, 2) << endl;

    return 0;
}

