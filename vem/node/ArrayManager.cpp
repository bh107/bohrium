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
	assert(base == NULL || base->base == NULL);

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

	if (base != NULL)
		base->ref_count++;

    return array;
}

ArrayManager::~ArrayManager()
{
    flush();
    delete arrayStore;
}

void ArrayManager::erase(cphvb_array* array)
{
	cphvb_array* base = cphvb_base_array(array);
	base->ref_count--;
	
	//We always delete views
	if (array->base != NULL)
	    arrayStore->erase(array);

	//If all views are accounted for, we delete the base view
	if (base->ref_count <= 0)
		arrayStore->erase(base);
}

void ArrayManager::erasePending(cphvb_instruction* inst)
{
    eraseQueue.push_back(inst);
}

void ArrayManager::changeOwnerPending(cphvb_instruction* inst, 
                                      cphvb_array* base,
                                      owner_t owner)
{
    assert(base->base == NULL);
    OwnerTicket t;
    t.array = base;
    t.instruction = inst;
    t.owner = owner;
    ownerChangeQueue.push_back(t);
}

void ArrayManager::flush()
{
    std::deque<OwnerTicket>::iterator oit = ownerChangeQueue.begin();

    //First we change ownership for all those pending
    for (; oit != ownerChangeQueue.end(); ++oit)
    {
    	if ((*oit).instruction->status == CPHVB_SUCCESS)
	        (*oit).array->owner = (*oit).owner;
    }
    // All ownerships are changed. So we clear the queue
    ownerChangeQueue.clear();

    //Then we delete arrays marked for deletion
    std::deque<cphvb_instruction*>::iterator eit = eraseQueue.begin();
    for (; eit != eraseQueue.end(); ++eit)
    {
    	if ((*eit)->status == CPHVB_SUCCESS)
	        this->erase((*eit)->operand[0]);
    }
    // All erases have been dealt with. So we clear the queue
    eraseQueue.clear();


}
