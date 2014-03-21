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
#include <iostream>
#include "bh/bh.hpp"
#include "util/timing.hpp"
#include "util/argparse.hpp"

using namespace std;
using namespace bh;
using namespace argparse;

template <typename T>
multi_array<T>& cnd(multi_array<T>& x)
{
    multi_array<T> l, k, w;
    multi_array<bh_bool> mask;
    T a1 = 0.31938153,
      a2 =-0.356563782,
      a3 = 1.781477937,
      a4 =-1.821255978,
      a5 = 1.330274429,
      pp = 2.5066282746310002; // sqrt(2.0*PI)

    l = abs(x);
    k = 1.0 / (1.0 + 0.2316419 * l);
    w = 1.0 - 1.0 / pp * exp(-1.0*l*l/2.0) * \
        (a1*k + \
         a2*(pow(k,(T)2)) + \
         a3*(pow(k,(T)3)) + \
         a4*(pow(k,(T)4)) + \
         a5*(pow(k,(T)5)));

    mask = x < 0.0;
    return w * as<T>(!mask) + (1.0-w)* as<T>(mask);
}

//FLOP count: 2*s+i*(s*8+2*s*26) where s is samples and i is iterations
template <typename T>
T* pricing(size_t samples, size_t iterations, char flag, T x, T d_t, T r, T v)
{
    multi_array<T> d1, d2, res, s;
    T* p    = (T*)malloc(sizeof(T)*iterations);    // Intermediate results
    T t     = d_t;                              // Initial delta

    s = random<T>((int64_t)samples)*4.0 +58.0;           // Model between 58-62

    for(size_t i=0; i<iterations; i++) {
        d1 = (log(s/x) + (r+v*v/2.0)*t) / (v*sqrt(t));
        d2 = d1-v*sqrt(t);

        if (flag == 'c') {
            res = s * cnd(d1) - x * exp(-r * t) * cnd(d2);
        } else {
            res = x * exp(-r*t) * cnd(-1.0*d2) - s*cnd(-1.0*d1);
        }

        t += d_t;                               // Increment delta
        p[i] = scalar(sum(res)) / (T)samples;   // Result from timestep
    }

    return p;
}

int main(int argc, char* argv[])
{
    const char usage[] = "usage: ./black_scholes --size=1000*10 [--verbose]";
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

    bh_intp start = _bh_timing();
    double* prices = pricing<double>(   // Do the computations...
        args.size[0], args.size[1],
        'c', 65.0, 1.0 / 365.0,
        0.08, 0.3
    );
                                        // Output timing
    cout << "{elapsed-time: "<< (_bh_timing()-start)/1000000.0 <<"";
    if (args.verbose) {                 // and values.
        cout << ", \"output\": [";
        for(size_t i=0; i<args.size[1]; i++) {
            cout << prices[i];
            if (args.size[1]-1!=i) {
                cout << ", ";
            }
        }
        cout << "]" << endl;
    }
    cout << "}" << endl;

    free(prices);                       // Cleanup
    return 0;
}

