/*
This file is part of Bohrium and copyright (c) 2012 the Bohrium
team <http://www.bh107.org>.

Bohrium is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as 
published by the Free Software Foundation, either version 3 
of the License, or (at your option) any later version.

Bohrium is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the 
GNU Lesser General Public License along with Bohrium. 

If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __SCALAR_HPP
#define __SCALAR_HPP

#include "KernelParameter.hpp"

class Scalar : public KernelParameter
{
private:
    OCLtype mytype;
    union value_t {
        cl_char c;
        cl_short s;
        cl_int i;
        cl_long l;
        cl_uchar uc;
        cl_ushort us;
        cl_uint ui;
        cl_ulong ul;
        // cl_half h;
        cl_float f;
        cl_double d;
        cl_float2 fcx;
        cl_double2 dcx;
        cl_ulong2 r123;
    } value;

protected:
    void printOn(std::ostream& os) const;
    void printValueOn(std::ostream& os) const;
public:
    Scalar(bh_base* spec);
    Scalar(bh_constant constant);
    Scalar(bh_intp);
    OCLtype type() const;
    void addToKernel(cl::Kernel& kernel, unsigned int argIndex);
    friend std::ostream& operator<<= (std::ostream& os, Scalar const& s);
};


#endif
