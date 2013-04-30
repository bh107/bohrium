#include <iostream>
#include "bh/bh.hpp"

using namespace std;
using namespace bh;

double monte_carlo_pi(int samples, int iterations)
{
    multi_array<double> x, y, m, c, accu(1), res(1);// Operands
    accu = (double)0.0;                             // Acculumate across iterations
    for(int i=0; i<iterations; ++i) {
        x = random<double>(samples);                // Sample random numbers
        y = random<double>(samples);
        m = (sqrt(x*x + y*y)<=1.0).as<double>();    // Model
        c = reduce(m, ADD, 0);                      // Count

        accu += (c*4.0) / (double)samples;          // Approximate
    }
    accu /= (double)iterations;
    
    return scalar(accu);
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
            ",iter="    << iterations << \
            ",val="     << monte_carlo_pi(samples, iterations) << \
            ")]"        << endl;

    stop();

    return 0;
}

