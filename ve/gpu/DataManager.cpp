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

void DataManager::initBuffer(cphVBarray* baseArray)
{
    assert(baseArray->base == NULL);
    if (baseArray->data == NULL)
    {
        if (baseArray->has_init_value)
        {
            //TODO memset now has to be implemented in the Kernel
            throw std::runtime_error("Setting initial value has not been imnplemented.");
        }
        else  
        {   //Nothing to init
            return;
        }
    }
    else
    {
        memoryManager->copyToDevice(baseArray);
    }
}

/* Map operands to GPU device pointers via base arrays.
 * Also updates cphVBarray with apropriate info.
 */
void DataManager::mapOperands(cphVBarray* operands[],
                              int nops)
{
    assert (nops > 0);
    Operand2BaseMap::iterator oiter;
    Base2BufferMap::iterator biter;
    cphVBarray* baseArray;
    cphVBarray* operand;
    for (int i = 0; i < nops; ++i)
    {
        operand = operands[i];
        if (operand != CPHVB_CONSTANT)
        {
#ifdef DEBUG
            std::cout << "[VE GPU] Mapping " << operand;
#endif            
            oiter = op2Base.find(operand);
            if (oiter == op2Base.end())
            {   //The operand is not mapped to a base array - so we will 
                baseArray = cphVBBaseArray(operand);
#ifdef DEBUG
            std::cout << " to " << baseArray;
#endif                            
                biter = base2Buffer.find(baseArray);
                if (biter == base2Buffer.end())
                {   //The base array is not mapped to a buffer - so we will
                    //We also need to initialize it
                    /* TODO: Do correct mapping 
                     * This works for now becaus we only support float32 
                     * and bool*/
                    baseArray->oclType = OCL_FLOAT32; //TODO HACK
                    cl::BUffer buffer = memoryManager->deviceAlloc(baseArray);
                    base2Buffer[baseArray] = buffer;
                    baseArray->buffer = buffer;
                    initBuffer(baseArray);
#ifdef DEBUG
                    std::cout << " which was unknown. Now has buffer " <<
                        (void*)buffer << std::endl;
#endif                            
                }
#ifdef DEBUG
                else
                {
                    std::cout << " which is known to have buffer " <<
                        (void*)baseArray->buffer << std::endl;
                }
#endif
                operand->buffer = baseArray->buffer;
                operand->oclType = baseArray->oclType;
                op2Base[operand] = baseArray;
            }

#ifdef DEBUG
            else
            {
                std::cout << ": Allready mapped to " << oiter->second << 
                    " with buffer " << (void*)oiter->second->buffer << 
                    std::endl;
            }
#endif
        }
    }
}

DataManager::DataManager(MemoryManager* memoryManager_) :
    memoryManager(memoryManager_),
    activeBatch(NULL) {}

void DataManager::lock(cphVBarray* operands[], 
                       int nops, 
                       InstructionBatch* batch)
{
    assert(nops > 0);
    mapOperands(operands, nops);
    cphVBarray* baseArray;

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
        for (int i = 1; i < nops; ++i)
        {
            _flush(operands[i]);
        }
    }
    /* Now we can just take the write lock on the array */
    baseArray = op2Base[operands[0]];
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
    
    // I may recieve sync for arrays I don't own 
    Base2CudaMap::iterator biter = base2Cuda.find(baseArray);
    if (biter == base2Buffer.end())
    {
       return;
    }
    _sync(baseArray);
    if (baseArray->data == NULL)
    {
        cphvb_data_ptr ptr = memoryManager->hostAlloc(baseArray);
        baseArray->data = ptr;
    }
    memoryManager->copyToHost(baseArray);
}

void DataManager::discard(cphVBarray* baseArray)
{
    assert(baseArray->base == NULL);
  
    // I may recieve discard for arrays I don't own
    Base2BufferMap::iterator biter = base2Buffer.find(baseArray);
    if (biter == base2Buffer.end())
    {
        return;
    }
    //TODO: Need to check if we need to flush: Is the array an input parameter 
    // for any operations
    flushAll();

    memoryManager->free(baseArray);
    baseArray->buffer = 0;
    base2Buffer.erase(baseArray);
    Operand2BaseMap::iterator oiter = op2Base.begin();
    while (oiter != op2Base.end())
    {
        if (oiter->second == baseArray)
        {
            op2Base.erase(oiter++);
        }
        else
        {
            ++oiter;
        }
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
