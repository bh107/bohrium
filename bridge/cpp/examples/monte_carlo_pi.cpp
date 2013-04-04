#include <iostream>
#include "bh/bh.hpp"

using namespace std;
using namespace bh;

multi_array<double>& monte_carlo_pi(int samples, int iterations)
{
    multi_array<double> x, y, m, c, sum(1);         // Operands

    sum = (double)0.0;                              // Acculumate across iterations
    for(int i=0; i<iterations; ++i) {
        x = random<double>(samples);                // Sample random numbers
        y = random<double>(samples);

        m = sqrt(x*x + y*y);                        // Model
        c = (m <= 1.0).as<double>().reduce(ADD,0);  // Count

        sum += (c*4.0) / (double)samples;           // Approximate
    }
    return sum / (double)iterations;                // Accumulated approximations
}

int main(int argc, char* argv[])
{
    int samples, iterations;
    if (argc < 3) {
        cout << "Error: Not enough argumnts, call like: " << endl;
        cout << argv[0] << " 1000 2" << endl;
        cout << "For 1000 samples and 2 iterations." << endl;
        return 0;
    }
    samples     = atoi(argv[1]);
    iterations  = atoi(argv[2]);

    cout << "[Pi Approximation (samples=" << samples << \
            ",iter=" <<iterations<< ")]" << endl;
    cout << monte_carlo_pi(samples, iterations) << endl;

    return 0;
}

