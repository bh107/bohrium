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

#ifndef __OFFSETMAP_HPP
#define __OFFSETMAP_HPP

#include <cphvb.h>
#include "cphVBarray.hpp"
#include "PTXregister.hpp"

class OffsetMap
{
public:
    virtual PTXregister* find(cphvb_intp ndim,
                              const cphvb_index shape[],
                              const cphvb_index stride[]) = 0;
    virtual PTXregister* find(const cphVBarray* array) = 0;
    virtual void insert(cphvb_intp ndim,
                        const cphvb_index shape[],
                        const cphvb_index stride[],
                        PTXregister* reg) = 0;
    virtual void insert(const cphVBarray* array,
                        PTXregister* reg) = 0;
    virtual void clear() = 0;
};


#endif
