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
#include "bh/bh.hpp"
using namespace bh;

#include "gtest/gtest.h"
#include "check_collections.hpp"

#define V_SIZE 3
#define M_SIZE 9
#define T_SIZE 27
const double res [] = {
    3.5,3.5,3.5, 3.5,3.5,3.5, 3.5,3.5,3.5
};

multi_array<float>& adder(float c)
{
    multi_array<float> a(3);

    a = (float)2.0;

    return a + c;
}

float resser(multi_array<float>& res)
{
    float val;
    multi_array<float> x(3);

    x = (float)5.0;
    x = adder(1.5);
    val = scalar(x);

    return val;
}

TEST(function__scope, external)
{

    float res = resser(adder(1.5));
    stop();

    EXPECT_TRUE(res == 3.5);
}


