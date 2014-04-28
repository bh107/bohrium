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
#include <complex>
#include <bh/bh.hpp>

using namespace bh;
using namespace std;

template <typename T>
void complex_ones()
{
    multi_array<std::complex<T> > cc;
    multi_array<T> r, i;

    cc      = ones<std::complex<T> >(3,3);
    r       = real<std::complex<T>, T>(cc);
    i       = imag<std::complex<T>, T>(cc);

    std::cout << r << "111" << i << "222" << cc << "!!!" << std::endl;
}

template <typename T>
void complex_constant()
{
    multi_array<std::complex<T> > cc;
    multi_array<T> r, i;

    cc      = ones<std::complex<T> >(3,3);
    cc      = (T)4.5;
    r       = real<std::complex<T>, T>(cc);
    i       = imag<std::complex<T>, T>(cc);

    std::cout << r << "111" << i << "222" << cc << "!!!" << std::endl;
}

void compute()
{
    std::cout << "Hello World." << std::endl;

    /*
    // Complex numbers
    complex_ones<double>();
    complex_ones<float>();

    complex_constant<double>();
    complex_constant<float>();
    */

    /*
    float* diller = (float*)malloc(100*sizeof(float));
    for(int i=0; i<100;++i) {
        diller[i] = i;
    }

    multi_array<float> hullet;
    hullet = empty<float>(100);
    hullet(diller);

    std::cout << "Hvad er der i hullet? " << hullet << "." << std::endl;
    std::cout << "Range(10)" << range<int64_t>(10) << std::endl;
    std::cout << "Range(0, 10, 2)" << range<int64_t>(0, 10, 1) << std::endl;

    1
    // Scan
    multi_array<float> a;
    a = ones<float>(20);
    a = scan(a, SUM, 0);

    std::cout << "a= 1,2,3, ... , 20:" << a << std::endl;
    std::cout << "prefix-sum(a): "  << scan(a, SUM, 0) << std::endl;
    std::cout << "prefix-prod(a): " << scan(a, PRODUCT, 0) << std::endl;

    multi_array<float> a;
    multi_array<float> b;

    // Transposition
    a = ones<float>(3,2,5);
    b = transpose(a);
    //b = sin(a);
    //b = a;
    std::cout << (b+a) << std::endl;

    //
    // Random
    cout << randomr<float>(10, 1, 0) << endl;
    cout << randomr<float>(10, 2, 0) << endl;
    */

    //
    // Explicit return...
    multi_array<float> a,b,c;
    a = ones<float>(10);
    b = ones<float>(10);
    c = ones<float>(10);

    cout << add(a, b) << endl;
    cout << add(c, a, b) << endl;

    c = add(a, b);
    cout << c << endl;

    add(c, a, b);
    cout << c << endl;

}

int main()
{
    compute();
    return 0;
}

