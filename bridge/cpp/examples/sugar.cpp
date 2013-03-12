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

using namespace bh;

template <typename T, int Dim0, int Dim1, int Dim2>
class stuff {
public:
    stuff();
};

template <typename T, int Dim0, int Dim1>
class stuff {
public:
    stuff();
};


template <typename T, int Dim0, 0>
class stuff {
public:
    stuff();
};


template <typename T, int Dim0, int Dim1>
stuff<T, Dim0, Dim1>::stuff() {
    if (Dim0>1)
        std::cout << "GGG " << Dim0 << std::endl;
    else
        std::cout << "poop " << Dim0 << std::endl;

}

// Same type different shape
template <typename T, int LDim0, int RDim0>
stuff<T, LDim0>& operator+(stuff<T, LDim0> &lhs, stuff<T, RDim0> &rhs)
{
    std::cout << "Different shape." << LDim0 << ", " << RDim0 << std::endl;
    stuff<T, LDim0>* operand = new stuff<T, LDim0>();
    return *operand;
}

template <typename T, int Dim0>
stuff<T, Dim0>& operator+(stuff<T, Dim0> &lhs, stuff<T, Dim0> &rhs)
{
    std::cout << "Same shape" << Dim0 << std::endl;
    stuff<T, Dim0>* operand = new stuff<T, Dim0>();
    return *operand;
}

void compute()
{
    stuff<double, 3> k = stuff<double,3>();
    stuff<double, 3> l = stuff<double,3>();
    k+l;

    stuff<double, 4> m = stuff<double,4>();
    m + l;

    multi_array<double> x(3);
    multi_array<double> y(9,3);
    multi_array<double> z;

    x = 2.0;
    y = 3.0;

    multi_array<double> s(9,3);

    std::cout << "Compatible? " << broadcast_shape(x, y, z) << "." << std::endl;
    s = x + y;
    pprint(s);
}

int main()
{
    std::cout << "Sugar?" << std::endl;

    compute();

    return 0;
}

