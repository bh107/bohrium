/*
This file is part of Bohrium and Copyright (c) 2012 the Bohrium team:
http://bohrium.bitbucket.org

Bohrium is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3
of the License, or (at your option) any later version.

Bohrium is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the
GNU Lesser General Public License along with bohrium.

If not, see <http://www.gnu.org/licenses/>.
*/
#include <inttypes.h>
#include <iostream>
#include <armadillo>
#include "util/timing.hpp"
#include "util/argparse.hpp"

using namespace std;
using namespace arma;
using namespace argparse;

template <typename T>
T compute_12_1a(size_t n, size_t iterations)
{
    T a, r;
    a = ones<T>(n, n);

    for(size_t i=0; i<iterations; i++) {
        r = a+a+a+a+a+a+a+a+a+a+a+a;
    }

    return r;
}

int main(int argc, char* argv[])
{
    const char usage[] = "usage: ./synth --size=1000*10 [--verbose]";
    if (2>argc) {
        cout << usage << endl;
        return 1;
    }

    arguments_t args;                   // Parse command-line
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

    uint64_t start = sample_time();
    fmat res = compute_12_1a<fmat>(args.size[0], args.size[1]);

                                        // Output timing
    cout << "{elapsed-time: "<< (sample_time()-start)/1000000.0 <<"";
    if (args.verbose) {                 // and values.
        cout << ", \"output\": [";
        cout << res << endl;;
        cout << "]" << endl;
    }
    cout << "}" << endl;

    return 0;
}
