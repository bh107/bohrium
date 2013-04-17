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
#include <cstdlib>
#include <cmath>
#include "bh/bh.hpp"

using namespace std;
using namespace bh;

template <typename T>
multi_array<T>& cnd(multi_array<T>& x)
{
    multi_array<T> L, K, w, mask;
    multi_array<bh_bool> mask_b;
    T a1 = 0.31938153,
      a2 =-0.356563782,
      a3 = 1.781477937,
      a4 =-1.821255978,
      a5 = 1.330274429,
      pp = 2.50662827463; // sqrt(2.0*PI)

    L = abs(x);
    K = 1.0 / (1.0 + 0.2316419 * L);
    w = 1.0 - 1.0 / (pp * exp(~L*L/2.0) * (a1*K + a2*(pow(K,(T)2)) + a3*(pow(K,(T)3)) + a4*(pow(K,(T)4)) + a5*(pow(K,(T)5))));

    mask_b = (x <= 0.0);
    mask = mask_b.as<T>();

    return w * ~mask + (1.0-w) * mask;
}

template <typename T>
multi_array<T>& black_scholes(multi_array<T>& s, char flag, T x, T u, T r, T v)
{
    multi_array<T> d1, d2, t1, t1_1, t2, t3, t4, top, top2;

    t1 = s/x;
    t1_1 = log(t1);
    t2 = (r+v*v/2.0);
    t3 = v*sqrt(u);
    cout << "1" << endl;
    top = t2 * u;
    cout << "2" << endl;
    top2 = top + t1;
    cout << "3" << endl;
    d1 = top2 / t3;
    cout << "4" << endl;

    cout << "5" << endl;
    t4 = d1 - v;
    cout << "6" << endl;
    d2 = t4 * sqrt(u);
    cout << "7" << endl;
    if (flag == 'c') {
        return s * cnd(d1) - x * exp(-1.0*r*u) * cnd(d2);
    } else {
        return x * exp(-r*u) * cnd(~d2) - s * cnd(~d1);
    }
}

template <typename T>
T* price(multi_array<T>& s, char flag, T x, T d_t, T r, T v, size_t iterations)
{
    size_t n = s.len();
    cout << "Alloc this much=" << sizeof(T)*n << "." << endl;
    T* p = (T*)malloc(sizeof(T)*iterations);

    cout << "w000t." << endl;
    T t = d_t;
    for(size_t i=0; i<iterations; i++) {    // Why sync after every iteration?
        cout << "asdfw000t." << endl;
        p[i] = *(black_scholes(s, flag, x, t, r, v).reduce(ADD,0).begin()) / (T)n;
        t += d_t;
    }
    return p;
}

template <typename T>
multi_array<T>& model(size_t& n)
{
    multi_array<T>& s = random<T>(n);
    s = s * 4.0 - 2.0 + 60.0; // Price is 58-62
    return s;
}

int main()
{
    size_t sample_size  = 1000,
           iterations   = 10;

    multi_array<double> s = model<double>(sample_size);
    double* prices = price(s, 'c', 65.0, 1.0/365.0, 0.08, 0.3, iterations);
    stop();

    cout << "Prices found: ";
    for(size_t i=0; i<iterations; i++) {
        cout << prices[i] << endl;
    }
    free(prices);

    return 0;
}

