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

#include <cstdio>
#include <stdexcept>
#include "PTXregister.hpp"

const char* prefix[] =
{
    /*[PTX_INT8] = */"sc",
    /*[PTX_INT16] = */"sh",
    /*[PTX_INT32] = */"si",
    /*[PTX_INT64] = */"sd",
    /*[PTX_UINT8] = */"uc",
    /*[PTX_UINT16] = */"uh",
    /*[PTX_UINT32] = */"ui",
    /*[PTX_UINT64] = */"ud",
    /*[PTX_FLOAT16] = */"fh",
    /*[PTX_FLOAT32] = */"f_",
    /*[PTX_FLOAT64] = */"fd",
    /*[PTX_BITS8] = */"bc",
    /*[PTX_BITS16] = */"bh",
    /*[PTX_BITS32] = */"b_",
    /*[PTX_BITS64] = */"bd",
    /*[PTX_PRED] = */"p"
};

int PTXregister::snprint(char* buf, int size)
{
    int res = std::snprintf(buf, size, "$%s%d", 
                            prefix[type],
                            typeIdx);
    if (res > size)
    {
        throw std::runtime_error("Not enough buffer space for printing.");
    }
    return res;
}
