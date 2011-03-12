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

#include <map>
#include <cphvb.h>
#include <iostream>
#include <sstream>
#include "PTXregister.hpp"
#include "OffsetMapSimple.hpp"

std::string toString(cphvb_intp ndim,
                     const cphvb_index shape[],
                     const cphvb_index stride[])
{
    std::ostringstream outs;
    for (int i = 0; i < ndim; ++i)
    {
        if (shape[i] != 1 && stride[i] != 0)
        {
            outs << shape[i] << stride[i];
        }
    }
    return outs.str();
}

OffsetMapSimple::OffsetMapSimple() {}

PTXregister* OffsetMapSimple::find(cphvb_intp ndim,
                                const cphvb_index shape[],
                                const cphvb_index stride[])
{
    std::string str = toString(ndim, shape, stride);
    MyOffsetMap::iterator iter = internalMap.find(str);
    if (iter == internalMap.end())
    {
        return NULL;
    }
    return iter->second;
}

PTXregister* OffsetMapSimple::find(const cphVBarray* array)
{
    return find(array->ndim, array->shape, array->stride);
}

void OffsetMapSimple::insert(cphvb_intp ndim,
                             const cphvb_index shape[],
                             const cphvb_index stride[],
                             PTXregister* reg)
{
    internalMap[toString(ndim, shape, stride)] = reg;
}

void OffsetMapSimple::insert(const cphVBarray* array,
                             PTXregister* reg)
{
    insert(array->ndim, array->shape, array->stride,reg);
}

void OffsetMapSimple::clear()
{
    internalMap.clear();
}

#ifdef DEBUG
PTXregister* OffsetMapSimple::get(int i)
{
    MyOffsetMap::iterator iter = internalMap.begin();
    while (i > 0)
    {
        ++iter; --i;
    }
    return iter->second;
}
#endif    

