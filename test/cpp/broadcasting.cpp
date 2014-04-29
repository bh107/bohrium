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


#include <stdexcept>
#include "gtest/gtest.h"
#include "check_collections.hpp"


#define V_SIZE 3
#define M_SIZE 9
#define T_SIZE 27
const double res [] = {
    3.5,3.5,3.5, 3.5,3.5,3.5, 3.5,3.5,3.5,
    3.5,3.5,3.5, 3.5,3.5,3.5, 3.5,3.5,3.5,
    3.5,3.5,3.5, 3.5,3.5,3.5, 3.5,3.5,3.5
};

TEST(broadcast,matrix_EQ_vector)
{
    multi_array<double> m(3,3);
    multi_array<double> v(3);

    v = 3.5;
    EXPECT_TRUE(CheckEqualCollections(v.begin(), v.end(), res));

    m = v;  /// This is the subject of the test
    EXPECT_TRUE(CheckEqualCollections(m.begin(), m.end(), res));
}

TEST(broadcast,vector_EQ_matrix)
{
    multi_array<double> m(3,3);
    multi_array<double> v(3);

    m = 3.5;
    v = m; /// This is the subject of the test

    EXPECT_TRUE(CheckEqualCollections(v.begin(), v.end(), res));
}

TEST(broadcast,matrix_EQ_matrix_plus_vector)
{
    multi_array<double> m(3,3);
    multi_array<double> v(3);
    multi_array<double> r(3,3);

    m = 2.0;
    v = 1.5;

    r = m+v;

    EXPECT_TRUE(CheckEqualCollections(r.begin(), r.end(), res));
}

TEST(broadcast,matrix_EQ_vector_plus_matrix)
{
    multi_array<double> m(3,3);
    multi_array<double> v(3);
    multi_array<double> r(3,3);

    m = 2.0;
    v = 1.5;

    r = v+m;

    EXPECT_TRUE(CheckEqualCollections(r.begin(), r.end(), res));
}

TEST(broadcast,tensor_EQ_vector)
{
    multi_array<double> t(3,3,3);
    multi_array<double> v(3);

    v = 3.5;
    t = v;

    EXPECT_TRUE(CheckEqualCollections(t.begin(), t.end(), res));
}

