#include <iostream>
#include "bxx/bohrium.hpp"
#include "util/timing.hpp"
#include "util/argparse.hpp"

using namespace std;
using namespace bh;
using namespace argparse;

double monte_carlo_pi(int samples, int iterations)
{
    multi_array<double> x, y, m, c, accu(1);        // Operands
    accu = (double)0.0;                             // Acculumate across iterations
    for(int i=0; i<iterations; ++i) {
        x = random<double>(samples);                // Sample random numbers
        y = random<double>(samples);
        m = as<double>(sqrt(x*x + y*y)<=1.0);    // Model
        c = reduce(m, ADD, 0);                      // Count

        accu += (c*4.0) / (double)samples;          // Approximate
    }
    accu /= (double)iterations;
    
    return scalar(accu);
}

int main(int argc, char* argv[])
{
    const char usage[] = "usage: ./monte_carlo_pi --size=1000*10 [--verbose]";
    if (2>argc) {
        cout << usage << endl;
        return 1;
    }

    arguments_t args;                               // Parse command-line
    if (!parse_args(argc, argv, args)) {
        cout << "Err: Invalid argument(s)." << endl;
        cout << usage << endl;
        return 1;
    }
    if (2 > args.size.size()) {
        cout << "Err: Not enough arguments." << endl;
        cout << usage << endl;
        return 1;
    }
    if (2 < args.size.size()) {
        cout << "Err: Too many arguments." << endl;
        cout << usage << endl;
        return 1;
    }

    size_t start = sample_time();
    double mcp = monte_carlo_pi(args.size[0], args.size[1]);
    size_t end = sample_time();
                                                    // Output timing
    cout << "{elapsed-time: "<< (end-start)/1000000.0 <<"";          
    if (args.verbose) {                             // and values.
        cout << ", \"output\": [";
        cout << mcp;
        cout << "]";
    }
    cout << "}" << endl;

    return 0;
}

