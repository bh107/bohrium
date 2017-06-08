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
#ifndef __BH_PPRINT_H
#define __BH_PPRINT_H

#include <iostream>
#include <vector>

#include <bh_opcode.h>
#include <bh_view.hpp>
#include <bh_ir.hpp>
#include <bh_type.hpp>


// Pretty print a C array
template<typename T>
std::string pprint_carray(const T *ary, uint64_t size) {
    std::stringstream ss;
    ss << "[";
    for (uint64_t i=0; i<size; ++i) {
        ss << ary[i];
        if (i+1 < size) {
            ss << ", ";
        }
    }
    ss << "]";
    return ss.str();
}

// Pretty print a std::vector
template<typename T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& v)
{
    out << "[";
    for (typename std::vector<T>::const_iterator i = v.cbegin();;)
    {
        out << *i;
        if (++i == v.cend())
            break;
        out << ", ";
    }
    out << "]";
    return out;
}

#endif
