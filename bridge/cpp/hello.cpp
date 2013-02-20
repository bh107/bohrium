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

void ac()
{
    Vector<double> x = Vector<double>(1024, 1024);
    x = 1.0;
}

void aa()
{
    Vector<double> x = Vector<double>(1024, 1024);
    Vector<double> y = Vector<double>(1024, 1024);

    x = 1.0;
    y = 2.0;

    x = y;
}

void aaa()
{
    Vector<double> x = Vector<double>(1024, 1024);
    Vector<double> y = Vector<double>(1024, 1024);
    Vector<double> z = Vector<double>(1024, 1024);

    x = 1.0;
    y = 2.0;
    z = 3.0;

    x = y + z;
}

void aac()
{
    Vector<double> x = Vector<double>(1024, 1024);
    Vector<double> y = Vector<double>(1024, 1024);
    double z;

    x = 1.0;
    y = 2.0;
    z = 3.0;

    x = y + z;
}

void aca()
{
    Vector<double> x = Vector<double>(1024, 1024);
    double y;
    Vector<double> z = Vector<double>(1024, 1024);

    x = 1.0;
    y = 2.0;
    z = 3.0;

    x = y + z;
}

void aa_u()
{
    Vector<double> x = Vector<double>(1024, 1024);
    Vector<double> y = Vector<double>(1024, 1024);

    y = 2.0;

    x = y;
}

void aaa_u()
{
    Vector<double> x = Vector<double>(1024, 1024);
    Vector<double> y = Vector<double>(1024, 1024);
    Vector<double> z = Vector<double>(1024, 1024);

    y = 2.0;
    z = 3.0;

    x = y + z;
}

void aac_u()
{
    Vector<double> x = Vector<double>(1024, 1024);
    Vector<double> y = Vector<double>(1024, 1024);
    double z;

    y = 2.0;
    z = 3.0;

    x = y + z;
}

void aca_u()
{
    Vector<double> x = Vector<double>(1024, 1024);
    double y;
    Vector<double> z = Vector<double>(1024, 1024);

    y = 2.0;
    z = 3.0;

    x = y + z;
}

void prefix()
{
    Vector<double> x = Vector<double>(1024, 1024);
    ++x;
}

void postfix()
{
    Vector<double> x = Vector<double>(1024, 1024);
    x++;
}

int main() {

    init();

    ac();
    aa();

    aaa();
    aac();
    aca();

    aa_u();
    aaa_u();
    aac_u();
    aca_u();

    prefix();
    postfix();

    shutdown();

    return 0;

}

