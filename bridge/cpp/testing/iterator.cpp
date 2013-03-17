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
#include "gtest/gtest.h"
#include "check_collections.hpp"

#include "bh/cppb.hpp"
using namespace bh;

#define V_SIZE 3
#define M_SIZE 9
#define T_SIZE 27
const double res [] = {
    3.5,3.5,3.5, 3.5,3.5,3.5, 3.5,3.5,3.5
};

TEST(iterator, vector)
{
    multi_array<double> x(V_SIZE);

    x = 3.5;

    /// This is the subject of the test
    EXPECT_TRUE(CheckEqualCollections(x.begin(), x.end(), res));
}

TEST(iterator, matrix)
{
    multi_array<double> x(3,3);

    x = 3.5;

    /// This is the subject of the test
    EXPECT_TRUE(CheckEqualCollections(x.begin(), x.end(), res));
}

TEST(iterator, tensor)
{
    multi_array<double> x(3,3,3);

    x = 3.5;

    /// This is the subject of the test
    EXPECT_TRUE(CheckEqualCollections(x.begin(), x.end(), res));
}


