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
#include "bxx/bohrium.hpp"
#include "util/timing.hpp"
#include "util/argparse.hpp"

using namespace std;
using namespace bxx;
using namespace argparse;

template <typename T>
void compute_add_4(uint64_t nelements, uint64_t iterations)
{
    multi_array<T> res;
    res = value<T>((T)1, nelements);

    Runtime::instance().flush();
    bh_intp start = sample_time();

    for(uint64_t i=0; i<iterations; ++i) {
        res = res + res + res + res + res + res + res + res + res + res + res;
        Runtime::instance().flush();
    }
    Runtime::instance().flush();
                                        // Output timing
    cout << "{elapsed-time: "<< (sample_time()-start)/1000000.0 <<"";
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

    compute_add_4<float>(args.size[0], args.size[1]);

    return 0;
}
