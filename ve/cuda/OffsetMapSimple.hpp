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

#ifndef __OFFSETMAPSIMPLE_HPP
#define __OFFSETMAPSIMPLE_HPP

#include <map>
#include <cphvb.h>
#include "cphVBarray.hpp"
#include "PTXregister.hpp"
#include "OffsetMap.hpp"

typedef std::map<std::string, PTXregister*> MyOffsetMap;

class OffsetMapSimple : public OffsetMap 
{
private:
    MyOffsetMap internalMap;
public:
    OffsetMapSimple();
    PTXregister* find(cphvb_intp ndim,
                      const cphvb_index shape[],
                      const cphvb_index stride[]);
    PTXregister* find(const cphVBarray* array);
    void insert(cphvb_intp ndim,
                const cphvb_index shape[],
                const cphvb_index stride[],
                PTXregister* reg);
    void insert(const cphVBarray* array,
                PTXregister* reg);
    void clear();
#ifdef DEBUG
    PTXregister* get(int i);
#endif    
};


#endif
