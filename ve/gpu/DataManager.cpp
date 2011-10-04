/*
 * Copyright 2011 Troels Blum <troels@blum.dk>
 *
 * This file is part of cphVB <http://code.google.com/p/cphvb/>.
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

#include <iostream>
#include <cassert>
#include <queue>
#include <map>
#include <cphvb.h>
#include "DataManager.hpp"
#include "WrapperFunctions.hpp"

void DataManager::_sync(cphVBarray* baseArray)
{
    if (activeBatch == NULL)
    {
        return;
    }
    WriteLockTable::iterator wliter = writeLockTable.find(baseArray);
    if (wliter != writeLockTable.end())
    {
        activeBatch->execute();
        writeLockTable.clear(); //OK because we are only working with one batch
    }
}

inline void DataManager::_flush(cphVBarray* view)
{
    if (activeBatch == NULL)
    {
        return;
    }
    cphVBarray* baseArray = op2Base[view];
    WriteLockTable::iterator wliter = writeLockTable.find(baseArray);
    if (wliter != writeLockTable.end() && wliter->second != view)
    {
        activeBatch->execute();
        writeLockTable.clear(); //OK because we are only working with one batch
    }
}

void DataManager::flush(cphVBarray* view)
{
    if (activeBatch == NULL)
    {
        return;
    }
    cphVBarray* baseArray = cphVBBaseArray(view);
    WriteLockTable::iterator wliter = writeLockTable.find(baseArray);
    if (wliter != writeLockTable.end())
    {
        activeBatch->execute();
        writeLockTable.clear(); //OK because we are only working with one batch
    }
}

DataManager::DataManager(ResourceManager* resourceManager_)
    : resourceManager(resourceManager_)
    , activeBatch(NULL) 
{}

void DataManager::lock(cphVBarray* operands[], 
                       int nops, 
                       InstructionBatch* batch)
{
    assert(nops > 0);
    for (int i = 0; i < nops; ++i)
    {
        cphvb_array* operand = operands[i];
        // Is it a new base array we haven't heard of before?
        if (operand != CPHVB_CONSTANT)
        {
            cphvb_array* base = cphvb_base_array(operand); 
            if (arrayMap.find(base) == arrayMap.end())
            {
                // Then create it
                arrayMap[base] = new BaseArray(base, resourceManager);
            }
        }
    }
    
    if (activeBatch == NULL)
    {
        activeBatch = batch;
    } 
    else if (batch != activeBatch)
    {
        activeBatch->execute();
        writeLockTable.clear(); //OK because we are only working with one batch
        activeBatch = batch;
    }
    else
    {
        /* We need to _flush all arrays that are read in the operation*/
        for (int i = 0; i < nops; ++i)
        {
            cphvb_array* operand = operands[i];
            if (operand != CPHVB_CONSTANT)
            {
                _flush(operands[i]);
            }
        }
    }
    /* Now we can just take the write lock on the array */
    cphVBarray* baseArray = op2Base[operands[0]];
    writeLockTable[baseArray] = operands[0];
}

void DataManager::release(cphVBarray* baseArray)
{
    sync(baseArray);
    discard(baseArray);
}

void DataManager::sync(cphVBarray* baseArray)
{
    assert(baseArray->base == NULL);
    // We may recieve sync for arrays I don't own
    ArrayMap::iterator it = arrayMap.find(base);
    if (it != arrayMap.end())
    {
        it->second->sync();
    }
}

void DataManager::discard(cphVBarray* baseArray)
{
    assert(baseArray->base == NULL);
    // We may recieve discard for arrays I don't own
    ArrayMap::iterator it = arrayMap.find(base);
    if (it != arrayMap.end())
    {
        //TODO: Need to check if we need to flush: Is the array an input parameter 
        // for any operations
        flushAll();
        arrayMap.erase(base);
    }
}

void DataManager::flushAll()
{
    if (activeBatch != NULL)
    {
        activeBatch->execute();
    }
    writeLockTable.clear(); //OK because we are only working with one batch
    activeBatch = NULL;
}

void DataManager::batchEnd()
{
#ifdef DEBUG
    std::cout << "[VE GPU] Datamanager::batchEnd() " << std::endl;
#endif
    flushAll();
}
