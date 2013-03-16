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
#define BOOST_TEST_MODULE basic_operations
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include "bh/cppb.hpp"
using namespace bh;

#define V_SIZE 3
const double res [] = { 3.5,3.5,3.5, 3.5,3.5,3.5, 3.5,3.5,3.5 };

/**
 * x = 3.5
 * Assignment Operator:
 * Assign a constant-value to every element of x.
 *
 */
BOOST_AUTO_TEST_CASE(const_assignment)
{
    multi_array<double> x(V_SIZE);

    BOOST_CHECK_NO_THROW( x = 3.5 );
    BOOST_CHECK_EQUAL_COLLECTIONS(x.begin(), x.end(), res, res+V_SIZE);
}

/**
 * x = x
 * Assignment Operator: this should be a noop!
 */
BOOST_AUTO_TEST_CASE(self_assignment)
{
    multi_array<double> x(V_SIZE);

    x = 3.5;

    int prior_k = keys;
    int prior_q = Runtime::instance()->queued();
    
    x = x;

    BOOST_CHECK_EQUAL(prior_k, keys);
    BOOST_CHECK_EQUAL(prior_q, Runtime::instance()->queued());
    BOOST_CHECK_EQUAL_COLLECTIONS(x.begin(), x.end(), res, res+V_SIZE);
}

/**
 * x = y
 * Assignment Operator:
 * Assign every element of y to to every element of x.
 *
 */
BOOST_AUTO_TEST_CASE(const_vector_assignment)
{
    multi_array<double> x(V_SIZE);
    multi_array<double> y(V_SIZE);

    BOOST_CHECK_NO_THROW( y = 3.5 );
    BOOST_CHECK_NO_THROW( x = y );
    BOOST_CHECK_EQUAL_COLLECTIONS(x.begin(), x.end(), res, res+V_SIZE);
}

/**
 * x = y
 * Assignment Operator:
 * Assign every element of y to to every element of x.
 *
 */
BOOST_AUTO_TEST_CASE(const_const_vector_assignment)
{
    multi_array<double> x(V_SIZE);
    multi_array<double> y(V_SIZE);

    BOOST_CHECK_NO_THROW(x = 1.5);
    BOOST_CHECK_NO_THROW(y = 3.5);
    BOOST_CHECK_NO_THROW(x = y);
    BOOST_CHECK_EQUAL_COLLECTIONS(x.begin(), x.end(), res, res+V_SIZE);
}

/**
 * x += k
 * Algebraic operator:
 * Add a constant value to every element of x.
 *
 */
BOOST_AUTO_TEST_CASE(compound_const_assignment)
{
    multi_array<double> x(V_SIZE);

    BOOST_CHECK_NO_THROW(x = 1.0);
    BOOST_CHECK_NO_THROW(x += 2.5);
    BOOST_CHECK_EQUAL_COLLECTIONS(x.begin(), x.end(), res, res+V_SIZE);
}

/**
 *  x += x
 *  Algebraic operator:
 *  Add every value of x to every element of x.
 *
 */
BOOST_AUTO_TEST_CASE(self_compound_vector_assignment)
{
    multi_array<double> x(V_SIZE);
    x = 1.75;

    BOOST_CHECK_NO_THROW(x += x)
    BOOST_CHECK_EQUAL_COLLECTIONS(x.begin(), x.end(), res, res+V_SIZE);
}

/**
 *  x += y
 *  Algebraic operator:
 *  Add every value of y to every element of x.
 *
 */
BOOST_AUTO_TEST_CASE(compound_vector_assignment)
{
    multi_array<double> x(V_SIZE);
    multi_array<double> y(V_SIZE);
    x = 1.0;
    y = 2.5;

    BOOST_CHECK_NO_THROW(x += y)
    BOOST_CHECK_EQUAL_COLLECTIONS(x.begin(), x.end(), res, res+V_SIZE);
}

/**
 *  x = x + y
 *  Algebraic operator:
 *  Add every value of y to every element of x.
 *
 */
BOOST_AUTO_TEST_CASE(binary_and_assignment)
{
    multi_array<double> x(V_SIZE);
    multi_array<double> y(V_SIZE);

    x = 1.0;
    y = 2.5;

    BOOST_CHECK_NO_THROW(x = x + y);
    BOOST_CHECK_EQUAL_COLLECTIONS(x.begin(), x.end(), res, res+V_SIZE);
}

/**
 *  x = y + x
 *  Algebraic operator:
 *  Add every value of x to every element of y.
 *
 */
BOOST_AUTO_TEST_CASE(xEQk_yWQk_xEQyx)
{
    multi_array<double> x(V_SIZE);
    multi_array<double> y(V_SIZE);

    x = 1.0;
    y = 2.5;

    BOOST_CHECK_NO_THROW(x = y + x);
    BOOST_CHECK_EQUAL_COLLECTIONS(x.begin(), x.end(), res, res+V_SIZE);
}

/**
 *  x = y + z
 *  Algebraic operator:
 *  Add every value of x to every element of y.
 *
 */
BOOST_AUTO_TEST_CASE(xEQk_yWQk_zEQk_xEQyx)
{
    multi_array<double> x(V_SIZE);
    multi_array<double> y(V_SIZE);
    multi_array<double> z(V_SIZE);

    x = 1.0;
    y = 1.0;
    z = 2.5;

    BOOST_CHECK_NO_THROW(x = y + z);
    BOOST_CHECK_EQUAL_COLLECTIONS(x.begin(), x.end(), res, res+V_SIZE);
}

/**
 *  x = y + z
 *  Algebraic operator:
 *  Add every value of x to every element of y.
 *
 */
BOOST_AUTO_TEST_CASE(yEQk_zEQk_xEQyz)
{
    multi_array<double> x(V_SIZE);
    multi_array<double> y(V_SIZE);
    multi_array<double> z(V_SIZE);

    y = 1.0;
    z = 2.5;

    x = y + z;
    BOOST_CHECK_NO_THROW(x = y + z);
    BOOST_CHECK_EQUAL_COLLECTIONS(x.begin(), x.end(), res, res+V_SIZE);
}

/**
 *  x = y + k
 *  Algebraic operator:
 *  Add every value of x to every element of y.
 *
 */
BOOST_AUTO_TEST_CASE(yEQk_xEQyk)
{
    multi_array<double> x(V_SIZE);
    multi_array<double> y(V_SIZE);

    y = 1.0;

    BOOST_CHECK_NO_THROW(x = y + 2.5);
    BOOST_CHECK_EQUAL_COLLECTIONS(x.begin(), x.end(), res, res+V_SIZE);
}

/**
 *  x = y + k
 *  Algebraic operator:
 *  Add every value of x to every element of y.
 *
 */
BOOST_AUTO_TEST_CASE(yEQk_xEQky)
{
    multi_array<double> x(V_SIZE);
    multi_array<double> y(V_SIZE);

    y = 1.0;

    BOOST_CHECK_NO_THROW(x = 2.5 + y);
    BOOST_CHECK_EQUAL_COLLECTIONS(x.begin(), x.end(), res, res+V_SIZE);
}

