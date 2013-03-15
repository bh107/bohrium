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
#define BOOST_TEST_MODULE construction
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include "bh/cppb.hpp"
using namespace bh;

#define CONTAINER_SIZE 9
const double res [] = { 3.5,3.5,3.5, 3.5,3.5,3.5, 3.5,3.5,3.5 };

BOOST_AUTO_TEST_CASE(assignment_vector_eq_const)
{
    multi_array<double> x(CONTAINER_SIZE);

    x = 3.5;    // The thing being tested...

    BOOST_CHECK_EQUAL_COLLECTIONS(x.begin(), x.end(), res, res+CONTAINER_SIZE);
}

BOOST_AUTO_TEST_CASE(assignment_vector_eq_vector)
{
    multi_array<double> x(9);
    multi_array<double> y(9);
    y = 3.5;

    x = y;      // The thing being tested...

    BOOST_CHECK_EQUAL_COLLECTIONS(x.begin(), x.end(), res, res+CONTAINER_SIZE);
}

BOOST_AUTO_TEST_CASE(assignment_matrix_eq_vector)
{
    multi_array<double> x(3,3);
    multi_array<double> y(3);
    y = 3.5;

    x = y;      // The thing being tested...

    BOOST_CHECK_EQUAL_COLLECTIONS(x.begin(), x.end(), res, res+CONTAINER_SIZE);
}

