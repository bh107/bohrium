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
#include <iostream>
#include <cassert>
#include "ArrayManager.hpp"
ArrayManager::ArrayManager() :
    arrayStore(new StaticStore<cphvb_array>(4096*10)) {}

cphvb_array* ArrayManager::create(cphvb_array* base,
                                  cphvb_type type,
                                  cphvb_intp ndim,
                                  cphvb_index start,
                                  cphvb_index shape[CPHVB_MAXDIM],
                                  cphvb_index stride[CPHVB_MAXDIM])
{
    cphvb_array* array = arrayStore->c_next();
    array->owner          = CPHVB_PARENT;
    array->base           = base;
    array->type           = type;
    array->ndim           = ndim;
    array->start          = start;
    array->data           = NULL;
    array->ref_count      = 1;
    std::memcpy(array->shape, shape, ndim * sizeof(cphvb_index));
    std::memcpy(array->stride, stride, ndim * sizeof(cphvb_index));

    if(array->base != NULL)
    {
        assert(array->base->base == NULL);
        ++array->base->ref_count;
        array->data = array->base->data;
    }
    return array;
}

ArrayManager::~ArrayManager()
{
    flush();
    delete arrayStore;
}

void ArrayManager::erasePending(cphvb_array* array)
{
    eraseQueue.push_back(array);
}

void ArrayManager::changeOwnerPending(cphvb_array* base,
                                      owner_t owner)
{
    assert(base->base == NULL);
    OwnerTicket t;
    t.array = base;
    t.owner = owner;
    ownerChangeQueue.push_back(t);
}

void ArrayManager::flush()
{
    std::deque<OwnerTicket>::iterator oit = ownerChangeQueue.begin();

    //First we change ownership for all those pending
    for (; oit != ownerChangeQueue.end(); ++oit)
    {
        (*oit).array->owner = (*oit).owner;
    }
    // All ownerships are changed. So we clear the queue
    ownerChangeQueue.clear();

    //Then we delete arrays marked for deletion
    std::deque<cphvb_array*>::iterator eit = eraseQueue.begin();
    for (; eit != eraseQueue.end(); ++eit)
    {
        if ((*eit)->base == NULL)
        {   //We have to deallocate the base array because of the
            //triggering opcode CPHVB_DESTROY.
            cphvb_data_free((*eit));
        }
        arrayStore->erase(*eit);
    }
    // All erases have been delt with. So we clear the queue
    eraseQueue.clear();
}
