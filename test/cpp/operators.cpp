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
const double res [] = { 3.5,3.5,3.5, 3.5,3.5,3.5, 3.5,3.5,3.5 };

/**
 * x = 3.5
 * Assignment Operator:
 * Assign a constant-value to every element of x.
 *
 */
TEST(operators,const_assignment)
{
    multi_array<double> x(V_SIZE);

    EXPECT_NO_THROW( x = 3.5 );
    EXPECT_TRUE(CheckEqualCollections(x.begin(), x.end(), res));
}

/**
 * x = x
 * Assignment Operator: this should be a noop!
 */
TEST(operators,self_assignment)
{
    multi_array<double> x(V_SIZE);

    x = 3.5;

    unsigned int prior_q = Runtime::instance().get_queue_size();
    
    x = x;

    EXPECT_EQ(prior_q, Runtime::instance().get_queue_size());
    EXPECT_TRUE(CheckEqualCollections(x.begin(), x.end(), res));
}

/**
 * x = y
 * Assignment Operator:
 * Assign every element of y to to every element of x.
 *
 */
TEST(operators,const_vector_assignment)
{
    multi_array<double> x(V_SIZE);
    multi_array<double> y(V_SIZE);

    EXPECT_NO_THROW( y = 3.5 );
    EXPECT_NO_THROW( x = y );
    EXPECT_TRUE(CheckEqualCollections(x.begin(), x.end(), res));
}

/**
 * x = y
 * Assignment Operator:
 * Assign every element of y to to every element of x.
 *
 */
TEST(operators,const_const_vector_assignment)
{
    multi_array<double> x(V_SIZE);
    multi_array<double> y(V_SIZE);

    EXPECT_NO_THROW(x = 1.5);
    EXPECT_NO_THROW(y = 3.5);
    EXPECT_NO_THROW(x = y);
    EXPECT_TRUE(CheckEqualCollections(x.begin(), x.end(), res));
}

/**
 * x += k
 * Algebraic operator:
 * Add a constant value to every element of x.
 *
 */
TEST(operators,compound_const_assignment)
{
    multi_array<double> x(V_SIZE);

    EXPECT_NO_THROW(x = 1.0);
    EXPECT_NO_THROW(x += 2.5);
    EXPECT_TRUE(CheckEqualCollections(x.begin(), x.end(), res));
}

/**
 *  x += x
 *  Algebraic operator:
 *  Add every value of x to every element of x.
 *
 */
TEST(operators,self_compound_vector_assignment)
{
    multi_array<double> x(V_SIZE);
    x = 1.75;

    EXPECT_NO_THROW(x += x);
    EXPECT_TRUE(CheckEqualCollections(x.begin(), x.end(), res));
}

/**
 *  x += y
 *  Algebraic operator:
 *  Add every value of y to every element of x.
 *
 */
TEST(operators,compound_vector_assignment)
{
    multi_array<double> x(V_SIZE);
    multi_array<double> y(V_SIZE);
    x = 1.0;
    y = 2.5;

    EXPECT_NO_THROW(x += y);
    EXPECT_TRUE(CheckEqualCollections(x.begin(), x.end(), res));
}

/**
 *  x = x + y
 *  Algebraic operator:
 *  Add every value of y to every element of x.
 *
 */
TEST(operators,binary_and_assignment)
{
    multi_array<double> x(V_SIZE);
    multi_array<double> y(V_SIZE);

    x = 1.0;
    y = 2.5;

    EXPECT_NO_THROW(x = x + y);
    EXPECT_TRUE(CheckEqualCollections(x.begin(), x.end(), res));
}

/**
 *  x = y + x
 *  Algebraic operator:
 *  Add every value of x to every element of y.
 *
 */
TEST(operators,xEQk_yWQk_xEQyx)
{
    multi_array<double> x(V_SIZE);
    multi_array<double> y(V_SIZE);

    x = 1.0;
    y = 2.5;

    EXPECT_NO_THROW(x = y + x);
    EXPECT_TRUE(CheckEqualCollections(x.begin(), x.end(), res));
}

/**
 *  x = y + z
 *  Algebraic operator:
 *  Add every value of x to every element of y.
 *
 */
TEST(operators,xEQk_yWQk_zEQk_xEQyx)
{
    multi_array<double> x(V_SIZE);
    multi_array<double> y(V_SIZE);
    multi_array<double> z(V_SIZE);

    x = 1.0;
    y = 1.0;
    z = 2.5;

    EXPECT_NO_THROW(x = y + z);
    EXPECT_TRUE(CheckEqualCollections(x.begin(), x.end(), res));
}

/**
 *  x = y + z
 *  Algebraic operator:
 *  Add every value of x to every element of y.
 *
 */
TEST(operators,yEQk_zEQk_xEQyz)
{
    multi_array<double> x(V_SIZE);
    multi_array<double> y(V_SIZE);
    multi_array<double> z(V_SIZE);

    y = 1.0;
    z = 2.5;

    x = y + z;
    EXPECT_NO_THROW(x = y + z);
    EXPECT_TRUE(CheckEqualCollections(x.begin(), x.end(), res));
}

/**
 *  x = y + k
 *  Algebraic operator:
 *  Add every value of x to every element of y.
 *
 */
TEST(operators,yEQk_xEQyk)
{
    multi_array<double> x(V_SIZE);
    multi_array<double> y(V_SIZE);

    y = 1.0;

    EXPECT_NO_THROW(x = y + 2.5);
    EXPECT_TRUE(CheckEqualCollections(x.begin(), x.end(), res));
}

/**
 *  x = y + k
 *  Algebraic operator:
 *  Add every value of x to every element of y.
 *
 */
TEST(operators,yEQk_xEQky)
{
    multi_array<double> x(V_SIZE);
    multi_array<double> y(V_SIZE);

    y = 1.0;

    EXPECT_NO_THROW(x = 2.5 + y);
    EXPECT_TRUE(CheckEqualCollections(x.begin(), x.end(), res));
}

