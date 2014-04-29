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
#include "bxx/bohrium.hpp"
using namespace bh;

#include "gtest/gtest.h"
#include "check_collections.hpp"

#define V_SIZE 3
#define M_SIZE 9
#define T_SIZE 27
const double res3D [] = {
    3.5,3.5,3.5, 3.5,3.5,3.5, 3.5,3.5,3.5,
    3.5,3.5,3.5, 3.5,3.5,3.5, 3.5,3.5,3.5,
    3.5,3.5,3.5, 3.5,3.5,3.5, 3.5,3.5,3.5
};
const double res2D [] = {
    10.5,10.5,10.5,
    10.5,10.5,10.5,
    10.5,10.5,10.5
};

const double res1D [] = {
    31.5, 31.5, 31.5
};

const double resS1 [] = {94.5};
const double resS = 94.5;

TEST(reduction, partial_3D_to_2D)
{
    multi_array<double> x(3,3,3), y;

    x = 3.5;
    y = reduce(x, ADD, 0.0);

    EXPECT_TRUE(CheckEqualCollections(y.begin(), y.end(), res2D));
}

TEST(reduction, partial_3D_to_1D)
{
    multi_array<double> x(3,3,3), y;

    x = 3.5;
    y = reduce(x, ADD, 0.0);
    y = reduce(y, ADD, 0.0);

    EXPECT_TRUE(CheckEqualCollections(y.begin(), y.end(), res1D));
}

TEST(reduction, partial_3D_to_0D)
{
    multi_array<double> x(3,3,3), y;

    x = 3.5;
    y = reduce(x, ADD, 0.0);
    y = reduce(y, ADD, 0.0);
    y = reduce(y, ADD, 0.0);

    EXPECT_TRUE(CheckEqualCollections(y.begin(), y.end(), resS1));
}

TEST(reduction, full____3D_to_S)
{
    multi_array<double> x(3,3,3);
    double res;

    x   = 3.5;
    res = scalar(sum(x));

    EXPECT_TRUE(res == 94.5);
}

TEST(reduction, full____2D_to_S)
{
    multi_array<double> x(3,3), y;
    double res;

    x   = 3.5;
    res = scalar(sum(x));

    EXPECT_TRUE(res == 31.5);
}

TEST(reduction, full____1D_to_S)
{
    multi_array<double> x(3), y;
    double res;

    x   = 3.5;
    res = scalar(sum(x));

    EXPECT_TRUE(res == 10.5);
}

