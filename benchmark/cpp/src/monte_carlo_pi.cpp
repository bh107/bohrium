#include <iostream>
#include "bh/bh.hpp"

using namespace std;
using namespace bh;

double monte_carlo_pi(int samples, int iterations)
{
    multi_array<double> x, y, m, c, accu(1);// Operands
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


    double mcp = monte_carlo_pi(samples, iterations);

    stop();

    return 0;
}

