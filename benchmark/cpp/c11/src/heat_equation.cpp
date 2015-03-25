#include <iostream>
#include <sstream>
#include <string>
#include <cmath>
#include "util/timing.hpp"
#include "util/argparse.hpp"

using namespace std;
using namespace argparse;

int main(int argc, char* argv[])
{
    const char usage[] = "usage: ./heat_equation --size=10000*10000*10 [--verbose]";
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
    if (args.size.size() < 2) {
        cout << "Err: Not enough arguments." << endl;
        cout << usage << endl;
        return 1;
    }
    if (args.size.size() > 3) {
        cout << "Err: Too many arguments." << endl;
        cout << usage << endl;
        return 1;
    }

    double epsilon  = 0.005;
    double delta    = epsilon+1.0;


    const int ydim = 10000;
    if (ydim != args.size[0]) {
        cout << "Multidimensional arrays in C11 does not support dynamic size, so it has to be: " << ydim << "." << endl;
        return 0;
    }

    const int xdim = 10000;
    if (xdim != args.size[1]) {
        cout << "Multidimensional arrays in C11 does not support dynamic size, so it has to be: " << xdim << "." << endl;
        return 0;
    }

    const int max_iterations = args.size[2];

    auto grid = new double[ydim][xdim];
    auto temp = new double[ydim][xdim];

    for(int i=0; i<ydim; i++){      // Initialize the grid
        for(int j=0;j<xdim;j++){
            grid[i][j] = 0;
        }
    }
    for(int i=0; i<ydim; i++){      // And borders
        grid[i][0]      = -273.15;
        grid[i][xdim-1] = -273.15;
    }
    for(int i=0; i<xdim; i++){
        grid[0][i]      = -273.15;
        grid[ydim-1][i] = 40.0;
    }

    size_t start = sample_time();
    auto iterations = 0;            // Compute the heat equation
    while(delta>epsilon) {
        ++iterations;
        delta = 0.0;
        for(int i=1; i<ydim-1; i++){
            #pragma omp parallel for reduction(+:delta)
            for(int j=1;j<xdim-1;j++){
                temp[i][j] = (grid[i-1][j] + grid[i+1][j] + grid[i][j] + grid[i][j-1] + grid[i][j+1])*0.2;
                delta += abs(temp[i][j] - grid[i][j]);
            }
        }

        #pragma omp parallel for collapse(2)
        for(int i=1;i<ydim-1; i++){
            for(int j=1;j<xdim-1;j++){
                grid[i][j] = temp[i][j];
            }
        }
        if (iterations>=max_iterations) {
            break;
        }
    }
    size_t end = sample_time();

    cout << "{elapsed-time: "<< (end-start)/1000000.0 <<"";          
    if (args.verbose) {                             // and values.
        cout << ", \"output\": [";
        for(int i=0; i<10; ++i) {
            for(int j=0; j<10; ++j) {
                cout << grid[i][j] << ", ";
            }
            cout << endl;
        }
        cout << "]";
    }
    cout << "}" << endl;

    return 0;
}
