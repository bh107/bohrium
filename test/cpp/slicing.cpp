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

TEST(slicing,vector_neg)
{
    multi_array<double> b(20);
    multi_array<double> v;

    bh_array* view = &storage[v.getKey()];

    b = 1.0;
    v = b[_(1,-1,2)];

    int shape[]     = {9};
    int stride[]    = {2};
    EXPECT_TRUE(VerifySlicing(view, shape, stride, 1, 1));
}

TEST(slicing,vector_neg2)
{
    multi_array<double> b(20);
    multi_array<double> v;

    bh_array* view = &storage[v.getKey()];

    b = 1.0;
    v = b[_(0,-2,1)];

    int shape[]     = {18};
    int stride[]    = {1};
    EXPECT_TRUE(VerifySlicing(view, shape, stride, 1, 0));
}

TEST(slicing,vector_even)
{
    multi_array<double> b(20);
    multi_array<double> v;

    bh_array* view = &storage[v.getKey()];

    b = 1.0;
    v = b[_(0,20,2)];

    int shape[]     = {10};
    int stride[]    = {2};
    EXPECT_TRUE(VerifySlicing(view, shape, stride, 1, 0));
}



TEST(slicing,vector_odd)
{
    multi_array<double> b(20);
    multi_array<double> v;

    bh_array* view = &storage[v.getKey()];

    b = 1.0;
    v = b[_(1,20,2)];

    int shape[]     = {10};
    int stride[]    = {2};
    EXPECT_TRUE(VerifySlicing(view, shape, stride, 1, 1));
}

TEST(slicing,vector_sub)
{
    multi_array<double> b(20);
    multi_array<double> v;

    bh_array* view = &storage[v.getKey()];

    b = 1.0;
    v = b[_(3,20,2)];

    int shape[]     = {9};
    int stride[]    = {2};
    EXPECT_TRUE(VerifySlicing(view, shape, stride, 1, 3));
}

TEST(slicing,matrix_A_AA2_inner_even)
{
    multi_array<double> b(9,9);
    multi_array<double> v;

    bh_array* view = &storage[v.getKey()];

    b = 1.0;
    v = b[_(0,9,1)][_(0,9,2)];

    int shape[]     = {9,5};
    int stride[]    = {9,2};
    EXPECT_TRUE(VerifySlicing(view, shape, stride, 2, 0));
}

TEST(slicing,matrix_AA2_A_outer_even)
{
    multi_array<double> b(9,9);
    multi_array<double> v;

    bh_array* view = &storage[v.getKey()];

    b = 1.0;
    v = b[_(0,9,2)][_(0,9,1)];

    int shape[]     = {5,9};
    int stride[]    = {18,1};
    EXPECT_TRUE(VerifySlicing(view, shape, stride, 2, 0));
}

TEST(slicing,matrix_AA2_AA2_both_even)
{
    multi_array<double> b(9,9);
    multi_array<double> v;

    bh_array* view = &storage[v.getKey()];

    b = 1.0;
    v = b[_(0,9,2)][_(0,9,2)];

    int shape[]     = {5,5};
    int stride[]    = {18,2};
    EXPECT_TRUE(VerifySlicing(view, shape, stride, 2, 0));

}

TEST(slicing,matrix_A_AA2_inner_odd)
{
    multi_array<double> b(9,9);
    multi_array<double> v;

    bh_array* view = &storage[v.getKey()];

    b = 1.0;
    v = b[_(0,9,1)][_(3,9,2)];

    int shape[]     = {9,3};
    int stride[]    = {9,2};
    EXPECT_TRUE(VerifySlicing(view, shape, stride, 2, 3));
}

TEST(slicing,matrix_AA2_A_outer_odd)
{
    multi_array<double> b(9,9);
    multi_array<double> v;

    bh_array* view = &storage[v.getKey()];

    b = 1.0;
    v = b[_(3,9,2)][_(0,9,1)];

    int shape[]     = {3,9};
    int stride[]    = {18,1};
    EXPECT_TRUE(VerifySlicing(view, shape, stride, 2, 27));
}

TEST(slicing,matrix_AA2_AA2_both_odd)
{
    multi_array<double> b(9,9);
    multi_array<double> v;

    bh_array* view = &storage[v.getKey()];

    b = 1.0;
    v = b[_(3,9,2)][_(3,9,2)];

    int shape[]     = {3,3};
    int stride[]    = {18,2};
    EXPECT_TRUE(VerifySlicing(view, shape, stride, 2, 30));
}

/*
TEST(slicing,assign_to_view_MATRIX2D)
{
    multi_array<double> b(9,9);

    bh_array* view = &storage[v.getKey()];

    b = 1.0;
    b[0][_(0,9,2)] = 3.5;
    b[-1][_(0,9,2)] = 3.5;

    int shape[]     = {3,3};
    int stride[]    = {18,2};
    EXPECT_TRUE(VerifySlicing(view, shape, stride, 2, 30));
}
*/
