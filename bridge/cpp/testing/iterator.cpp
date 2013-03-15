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
#define BOOST_TEST_MODULE iterator
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <stdexcept>

#include "bh/cppb.hpp"
using namespace bh;

#define V_SIZE 3
#define M_SIZE 9
#define T_SIZE 27
const double res [] = {
    3.5,3.5,3.5, 3.5,3.5,3.5, 3.5,3.5,3.5
};

BOOST_AUTO_TEST_CASE(vector_iter)
{
    multi_array<double> x(V_SIZE);

    x = 3.5;

    /// This is the subject of the test
    BOOST_REQUIRE_EQUAL_COLLECTIONS(x.begin(), x.end(), res, res+V_SIZE);
}

BOOST_AUTO_TEST_CASE(matrix_iter)
{
    multi_array<double> x(3,3);

    x = 3.5;

    /// This is the subject of the test
    BOOST_REQUIRE_EQUAL_COLLECTIONS(x.begin(), x.end(), res, res+M_SIZE);
}

BOOST_AUTO_TEST_CASE(tensor_iter)
{
    multi_array<double> x(3,3,3);

    x = 3.5;

    /// This is the subject of the test
    BOOST_REQUIRE_EQUAL_COLLECTIONS(x.begin(), x.end(), res, res+T_SIZE);
}


