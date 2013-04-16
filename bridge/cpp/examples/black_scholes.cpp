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
    multi_array<T> L, K, w, mask;
    T a1 = 0.31938153,
      a2 =-0.356563782,
      a3 = 1.781477937,
      a4 =-1.821255978,
      a5 = 1.330274429,
      pp = 2.50662827463; // sqrt(2.0*PI)

    L = abs(x);
    K = 1.0 / (1.0 + 0.2316419 * L);
    w = 1.0 - 1.0 / (pp * exp(~L*L/2.0) * (a1*K + a2*(pow(K,2)) + a3*(pow(K,3)) + a4*(pow(K,4)) + a5*(pow(K,5))));

    mask    = x < 0;
    w       = w * ~mask + (1.0-w)*mask;

    return w;
}

template <typename T>
multi_array<T>& black_scholes(multi_array<T>& S, char flag, T X, T U, T r, T v)
{
    multi_array<T> d1, d2;

    d1 = (log(S/X)+(r+v*v/2.0)*U)/(v*sqrt(U));
    d2 = d1-v*sqrt(U);
    if (flag == 'c') {
        return S * cnd(d1) - X * exp(-1.0 * r * U) * cnd(d2);
    } else {
        return X * exp(~r*U) * cnd(~d2) - S*cnd(~d1);
    }
}

template <typename T>
T* price(multi_array<T>& S, char flag, T X, T dU, T r, T v, size_t iterations)
{
    T U = dU;
    T N = (T)S.len();
    T* p;
    p = new T[N];

    for(size_t i=0; i<iterations; i++) {
        p[i] = (sum(black_scholes(flag, S, X, U, r, v)) / N).first();
        U += dU;
    }
    return p;
}

template <typename T>
multi_array<T>& model(size_t& n)
{
    multi_array<T>& s;
    s = random<T>(n);
    s = s * 4.0 - 2.0 + 60.0; // Price is 58-62
    return s;
}

int main()
{
    size_t sample_size  = 1000,
           iterations   = 10;

    multi_array<double> S;
    S = model(sample_size);
    double* prices = price(S, 'c', 65.0, 1.0/365.0, 0.08, 0.3, iterations);
    stop();

    cout << "Prices found: ";
    for(size_t i=0; i<iterations; i++) {
        cout << prices[i] << endl;
    }
    delete prices;
    return 0;
}

