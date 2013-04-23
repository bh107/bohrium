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

using namespace std;
using namespace bh;

template <typename T>
multi_array<T>& cnd(multi_array<T>& x)
{
    multi_array<T> l, k, w, mask;
    T a1 = 0.31938153,
      a2 =-0.356563782,
      a3 = 1.781477937,
      a4 =-1.821255978,
      a5 = 1.330274429,
      pp = 2.50662827463; // sqrt(2.0*PI)

    l = abs(x);
    k = 1.0 / (1.0 + 0.2316419 * l);
    w = 1.0 - 1.0 / (pp * exp(~l*l/2.0) * \
        (a1*k + \
         a2*(pow(k,(T)2)) + \
         a3*(pow(k,(T)3)) + \
         a4*(pow(k,(T)4)) + \
         a5*(pow(k,(T)5))
        )
    );

    mask = (x<0.0).template as<T>();
    //return w * ~mask + (1.0-w)*mask;
    return w * mask + (1.0-w)*mask;
}

template <typename T>
multi_array<T>& black_scholes(multi_array<T>& s, char flag, T x, T t, T r, T v)
{
    multi_array<T> d1, d2;

    d1 = (log(s/x) + (r+v*v/2.0)*t) / (v*sqrt(t));
    d2 = d1-v*sqrt(t);

    if (flag == 'c') {
        return s * cnd(d1) - x * exp(-1.0 * r * t) * cnd(d2);
    } else {
        //return x * exp(~r*t) * cnd(~d2) - s*cnd(~d1);
        return x * exp(r*t) * cnd(d2) - s*cnd(d1);
    }
}

template <typename T>
T* price(multi_array<T>& s, char flag, T x, T d_t, T r, T v, size_t iterations)
{
    T t = d_t;
    size_t n = (T)s.len();
    T* p;
    p = new T[n];

    for(size_t i=0; i<iterations; i++) {
        p[i] = *(black_scholes(s, flag, x, d_t, r, v).sum().begin()) / (T)n;
        t += d_t;
    }

    return p;
}


int main()
{
    size_t sample_size  = 1000,
           iterations   = 10;

    multi_array<double> s;

    s = random<double>(sample_size) * (-2.0) + 60.0; // Model price between 58-62

    double* prices = price(s, 'c', 65.0, 1.0/365.0, 0.08, 0.3, iterations);
    stop();

    cout << "Prices found: ";
    for(size_t i=0; i<iterations; i++) {
        cout << prices[i] << endl;
    }
    delete prices;
    return 0;
}

