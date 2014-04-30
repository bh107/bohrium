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
using namespace bxx;


#include "gtest/gtest.h"
#include "check_collections.hpp"


#define V_SIZE 9
#define M_SIZE 9
#define T_SIZE 27
const double res [] = {
    3.5,3.5,3.5, 3.5,3.5,3.5, 3.5,3.5,3.5
};

TEST(compound,vector_ADD_EQ_const)
{
    multi_array<double> x(9);

    x = 2.0;
    x += 1.5;   /// This is the subject of the test

    EXPECT_TRUE(CheckEqualCollections(x.begin(), x.end(), res));
}

TEST(compound,vector_ADD_EQ_vector)
{
    multi_array<double> x(9);
    multi_array<double> y(9);

    x = 2.0;
    y = 1.5;

    x += y;   /// This is the subject of the test

    EXPECT_TRUE(CheckEqualCollections(x.begin(), x.end(), res));
}
