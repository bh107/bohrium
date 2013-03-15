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
#define BOOST_TEST_MODULE broadcast
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <stdexcept>

#include "bh/cppb.hpp"
using namespace bh;

#define V_SIZE 3
#define M_SIZE 9
#define T_SIZE 27
const double res [] = {
    3.5,3.5,3.5, 3.5,3.5,3.5, 3.5,3.5,3.5,
    3.5,3.5,3.5, 3.5,3.5,3.5, 3.5,3.5,3.5,
    3.5,3.5,3.5, 3.5,3.5,3.5, 3.5,3.5,3.5
};

BOOST_AUTO_TEST_CASE(matrix_EQ_vector)
{
    multi_array<double> m(3,3);
    multi_array<double> v(3);

    v = 3.5;
    BOOST_REQUIRE_EQUAL_COLLECTIONS(v.begin(), v.end(), res, res+V_SIZE);

    m = v;  /// This is the subject of the test
    BOOST_REQUIRE_EQUAL_COLLECTIONS(m.begin(), m.end(), res, res+M_SIZE);
}

BOOST_AUTO_TEST_CASE(vector_EQ_matrix)
{
    multi_array<double> m(3,3);
    multi_array<double> v(3);

    m = 3.5;
    BOOST_CHECK_EQUAL_COLLECTIONS(m.begin(), m.end(), res, res+M_SIZE);

    BOOST_REQUIRE_THROW( v=m, std::runtime_error ); /// This is the subject of the test
}

BOOST_AUTO_TEST_CASE(matrix_EQ_matrix_plus_vector)
{
    multi_array<double> m(3,3);
    multi_array<double> v(3);
    multi_array<double> r(3,3);

    m = 2.0;
    v = 1.5;

    r = m+v;

    BOOST_CHECK_EQUAL_COLLECTIONS(r.begin(), r.end(), res, res+M_SIZE);
}

BOOST_AUTO_TEST_CASE(matrix_EQ_vector_plus_matrix)
{
    multi_array<double> m(3,3);
    multi_array<double> v(3);
    multi_array<double> r(3,3);

    m = 2.0;
    v = 1.5;

    r = v+m;

    BOOST_CHECK_EQUAL_COLLECTIONS(r.begin(), r.end(), res, res+M_SIZE);
}

BOOST_AUTO_TEST_CASE(tensor_EQ_vector)
{
    multi_array<double> t(3,3,3);
    multi_array<double> v(3);

    v = 3.5;
    t = v;

    BOOST_CHECK_EQUAL_COLLECTIONS(t.begin(), t.end(), res, res+T_SIZE);
}

