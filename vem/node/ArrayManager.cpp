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

#include <cstring>
#include <cassert>
#include <cphvb_vem_node.h>
#include "ArrayManager.hpp"
ArrayManager::ArrayManager() :
    arrayStore(new StaticStore<cphvb_array>(4096)) {}

cphvb_array* ArrayManager::create(cphvb_array* base,
                                  cphvb_type type,
                                  cphvb_intp ndim,
                                  cphvb_index start,
                                  cphvb_index shape[CPHVB_MAXDIM],
                                  cphvb_index stride[CPHVB_MAXDIM],
                                  cphvb_intp has_init_value,
                                  cphvb_constant init_value)
{
    cphvb_array* array = arrayStore->c_next();
    
    array->owner          = CPHVB_PARENT;
    array->base           = base;
    array->type           = type;
    array->ndim           = ndim;
    array->start          = start;
    array->has_init_value = has_init_value;
    array->init_value     = init_value;
    array->data           = NULL;
    array->ref_count      = 1;
    std::memcpy(array->shape, shape, ndim * sizeof(cphvb_index));
    std::memcpy(array->stride, stride, ndim * sizeof(cphvb_index));

    if(array->base != NULL)
    {
        assert(array->base->base == NULL);
        assert(!has_init_value);
        ++array->base->ref_count;
        array->data = array->base->data;
    }
    return array;
}

void ArrayManager::erasePending(cphvb_array* array)
{
    eraseQueue.push_back(array);
}

void ArrayManager::changeOwnerPending(cphvb_array* base,
                                      cphvb_comp owner)
{
    assert(base->base == NULL);
    ownerChangeQueue.push_back((OwnerTicket){base,owner});
}

void ArrayManager::flush()
{
    std::pair<BaseToViewMMap::iterator,BaseToViewMMap::iterator> range;
    BaseToViewMMap::iterator rit;
    std::deque<OwnerTicket>::iterator oit = ownerChangeQueue.begin();

    //First we change ownership for all those pending
    for (; oit != ownerChangeQueue.end(); ++oit)
    {
        (*oit).array->owner = (*oit).owner;
        if ((*oit).array->base == NULL)
        {   //if it is a base array we chech to se if there are stale views
            range = staleView.equal_range((*oit).array);
            for (rit=range.first; rit!=range.second; ++rit)
            {   //If there are any we delete them
                arrayStore->erase(rit->second);
                staleView.erase(rit);
            }
        }
    }
    // All ownerships have been changes. So we clear the queue
    ownerChangeQueue.clear();

    //Then we delete those array specs marked for deletion
    std::deque<cphvb_array*>::iterator eit = eraseQueue.begin();
    for (; eit != eraseQueue.end(); ++eit)
    {
        if ((*eit)->base == NULL)
        {   //Its a base array so we can delete it
            arrayStore->erase(*eit);
        }
        else if ((*eit)->base->owner == CPHVB_PARENT)
        {   //The base owner is up stream so we can delete it
            arrayStore->erase(*eit);
        }
        else
        {   //The base owner is down stream. So we postpone deletion
            staleView.insert(std::pair<cphvb_array*, 
                                 cphvb_array*>((*eit)->base, *eit));
        }
    }
    // All erases have been delt with. So we clear the queue
    eraseQueue.clear();
}
