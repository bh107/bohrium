/*
 * Copyright 2011 Troels Blum <troels@blum.dk>
 *
 * This file is part of cphVB.
 *
 * cphVB is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * cphVB is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with cphVB.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <cassert>
#include <cstdio>
#include <stdexcept>
#include "PTXconstant.hpp"

int PTXconstant::snprint(char* buf, int size)
{
    int res;
    switch(type)
    {
    case PTX_INT:
        res = std::snprintf(buf, size, "%ld", value.i);
        break;
    case PTX_UINT:
        res = std::snprintf(buf, size, "%ldU", value.u);
        break;
    case PTX_FLOAT:
        res = std::snprintf(buf, size, "%#.24e", value.f);
        break;
    case PTX_BITS:
        res = std::snprintf(buf, size, "%p", (void*)value.a);
        break;
    default:
        assert(false);
    }
    if (res > size)
    {
        throw std::runtime_error("Not enough buffer space for printing.");
    }
    return res;
}

