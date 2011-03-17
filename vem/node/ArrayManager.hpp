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

#ifndef __ARRAYMANAGER_HPP
#define __ARRAYMANAGER_HPP

#include <map>
#include <deque>
#include <cphvb.h>
#include <StaticStore.hpp>

typedef std::multimap<cphvb_array*, cphvb_array*> ViewMap;

class ArrayManager
{
private:
    StaticStore<cphvb_array>* arrayStore;
    ViewMap deletePending; // for delete pending accross batches
    std::deque<cphvb_array*> eraseQueue;
    std::deque<cphvb_array*> ownerChangeQueue;
    
public:
    ArrayManager();
    cphvb_array* create(cphvb_array* base,
                        cphvb_type type,
                        cphvb_intp ndim,
                        cphvb_index start,
                        cphvb_index shape[CPHVB_MAXDIM],
                        cphvb_index stride[CPHVB_MAXDIM],
                        cphvb_intp has_init_value,
                        cphvb_constant init_value);
    void erasePending(cphvb_array* array);
    void changeOwnerPending(cphvb_array* base,
                            cphvb_intp owner);
    void flush();
};


#endif
