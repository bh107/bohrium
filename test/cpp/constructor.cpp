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


#define V_SIZE 3
#define M_SIZE 9
#define T_SIZE 27
const double res [] = {
    3.5,3.5,3.5, 3.5,3.5,3.5, 3.5,3.5,3.5
};

TEST(constructor, copy_I)
{
    /// This is the subject of the test
    multi_array<double> x = multi_array<double>(V_SIZE);

    x = 3.5;
    
    EXPECT_TRUE(CheckEqualCollections(x.begin(), x.end(), res));
}

TEST(constructor, copy_I_I)
{
    /// This is the subject of the test
    multi_array<double> x = multi_array<double>(V_SIZE, V_SIZE);

    x = 3.5;
    
    EXPECT_TRUE(CheckEqualCollections(x.begin(), x.end(), res));
}

TEST(constructor, regular_I)
{
    /// This is the subject of the test
    multi_array<double> x(V_SIZE);

    x = 3.5;
    
    EXPECT_TRUE(CheckEqualCollections(x.begin(), x.end(), res));
}

TEST(constructor, regular_I_I)
{
    /// This is the subject of the test
    multi_array<double> x(V_SIZE, V_SIZE);

    x = 3.5;
    
    EXPECT_TRUE(CheckEqualCollections(x.begin(), x.end(), res));
}

