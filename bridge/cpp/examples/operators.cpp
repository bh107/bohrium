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
#include <iostream>
#include "bh_cppb.hpp"

using namespace bh;

/**
 * x = 1.0
 * Assignment Operator:
 * Assign a constant-value to every element of x.
 *
 */
void xk()
{
    Vector<double> x = Vector<double>(1024, 1024);

    x = 1.0;
}

/**
 * x = x
 * Assignment Operator:
 * Assign every element of x to to every element of x.
 *
 */
void xk_xx()
{
    Vector<double> x = Vector<double>(1024, 1024);

    x = 1.0;

    x = x;
}

/**
 * x = y
 * Assignment Operator:
 * Assign every element of y to to every element of x.
 *
 */
void yk_xy()
{
    Vector<double> x = Vector<double>(1024, 1024);
    Vector<double> y = Vector<double>(1024, 1024);

    y = 2.0;

    x = y;
}

/**
 * x = y
 * Assignment Operator:
 * Assign every element of y to to every element of x.
 *
 */
void xk_yk_xy()
{
    Vector<double> x = Vector<double>(1024, 1024);
    Vector<double> y = Vector<double>(1024, 1024);

    x = 1.0;
    y = 2.0;

    x = y;
}

/**
 * x += k
 * Algebraic operator:
 * Add a constant value to every element of x.
 *
 */
void xk_xxk()
{
    Vector<double> x = Vector<double>(1024, 1024);

    x = 1.0;

    x += 2.0;
}

/**
 *  x += x
 *  Algebraic operator:
 *  Add every value of x to every element of x.
 *
 */
void xk_xxx()
{
    Vector<double> x = Vector<double>(1024, 1024);
    x = 1.0;

    x += x;
}

/**
 *  x += y
 *  Algebraic operator:
 *  Add every value of y to every element of x.
 *
 */
void xk_yk_xxy()
{
    Vector<double> x = Vector<double>(1024, 1024);
    Vector<double> y = Vector<double>(1024, 1024);
    x = 1.0;
    y = 1.0;

    x += y;
}

/**
 *  x = x + y
 *  Algebraic operator:
 *  Add every value of y to every element of x.
 *
 */
void xk_yk_xxy_b()
{
    Vector<double> x = Vector<double>(1024, 1024);
    Vector<double> y = Vector<double>(1024, 1024);

    x = 1.0;
    y = 1.0;

    x = x + y;
}

/**
 *  x = y + x
 *  Algebraic operator:
 *  Add every value of x to every element of y.
 *
 */
void xk_yk_xyx_b()
{
    Vector<double> x = Vector<double>(1024, 1024);
    Vector<double> y = Vector<double>(1024, 1024);

    x = 1.0;
    y = 1.0;

    x = y + x;
}

/**
 *  x = y + z
 *  Algebraic operator:
 *  Add every value of x to every element of y.
 *
 */
void xk_yk_zk_xyz_b()
{
    Vector<double> x = Vector<double>(1024, 1024);
    Vector<double> y = Vector<double>(1024, 1024);
    Vector<double> z = Vector<double>(1024, 1024);

    x = 1.0;
    y = 1.0;
    z = 1.0;

    x = y + z;
}

/**
 *  x = y + z
 *  Algebraic operator:
 *  Add every value of x to every element of y.
 *
 */
void yk_zk_xyz_b()
{
    Vector<double> x = Vector<double>(1024, 1024);
    Vector<double> y = Vector<double>(1024, 1024);
    Vector<double> z = Vector<double>(1024, 1024);

    y = 1.0;
    z = 1.0;

    x = y + z;
}

/**
 *  x = y + k
 *  Algebraic operator:
 *  Add every value of x to every element of y.
 *
 */
void yk_xyk_b()
{
    Vector<double> x = Vector<double>(1024, 1024);
    Vector<double> y = Vector<double>(1024, 1024);

    y = 1.0;

    x = y + 2.0;
}

/**
 *  x = y + k
 *  Algebraic operator:
 *  Add every value of x to every element of y.
 *
 */
void yk_xky_b()
{
    Vector<double> x = Vector<double>(1024, 1024);
    Vector<double> y = Vector<double>(1024, 1024);

    y = 1.0;

    x = 2.0 + y;
}

int main() {

    init();

    xk();
    xk_xx();
    yk_xy();
    xk_xxk();
    xk_xxx();
    xk_yk_xy();
    xk_yk_xxy();

    yk_xky_b();
    yk_xyk_b();

    xk_yk_xxy_b();
    xk_yk_xyx_b();

    yk_zk_xyz_b();
    xk_yk_zk_xyz_b();

    shutdown();

    return 0;

}

