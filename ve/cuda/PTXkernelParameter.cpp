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
 * along with cphVB. If not, see <http://www.gnu.org/licenses/>.
 */

#include <cstdio>
#include <stdexcept>
#include "PTXkernelParameter.hpp"

int PTXkernelParameter::declare(const char* prefix, char* buf, int size)
{
    int res = std::snprintf(buf, size, "%s.param %s kp%ld_",
                            prefix, ptxTypeStr(type), id);
    if (res > size)
    {
        throw std::runtime_error("Not enough buffer space for printing.");
    }
    return res;
}

int PTXkernelParameter::declare(char* buf, int size)
{
    return declare("",buf,size);
}

int PTXkernelParameter::snprint(const char* prefix, 
                                char* buf, 
                                int size, 
                                const char* postfix)
{
    int res = std::snprintf(buf, size, "%skp%ld_%s", prefix, id, postfix);
    if (res > size)
    {
        throw std::runtime_error("Not enough buffer space for printing.");
    }
    return res;
}
