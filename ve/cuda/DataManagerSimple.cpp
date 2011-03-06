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

#include <iostream>
#include <cassert>
#include <queue>
#include <map>
#include <cphvb.h>
#include "DataManagerSimple.hpp"
#include "WrapperFunctions.hpp"

void DataManagerSimple::_sync(cphVBArray* baseArray)
{
    WriteLockTable::iterator wliter = writeLockTable.find(baseArray);
    if (wliter != writeLockTable.end())
    {
        activeBatch->execute();
        activeBatch = NULL;
    }
}

void DataManagerSimple::initCudaArray(cphVBArray* baseArray)
{
    assert(baseArray->base == NULL);
    if (baseArray->data == NULL)
    {
        if (baseArray->has_init_value)
        {
            memoryManager->memset(baseArray);
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

/* Map operands to CUDA device pointers via base arrays.
 * Also updates cphVBArray with apropriate info.
 */
void DataManagerSimple::mapOperands(cphVBArray* operands[],
                                    int nops)
{
    assert (nops > 0);
    Operand2BaseMap::iterator oiter;
    Base2CudaMap::iterator biter;
    cphVBArray* baseArray;
    cphVBArray* operand;
    for (int i = 0; i < nops; ++i)
    {
        operand = operands[i];
        oiter = op2Base.find(operand);
        if (oiter == op2Base.end())
        {   //The operand is not mapped to a base array - so we will 
            baseArray = cphVBBaseArray(operand);
            biter = base2Cuda.find(baseArray);
            if (biter == base2Cuda.end())
            {   //The base array is not mapped to a cudaPtr - so we will
                //We also need to initialize it
                CUdeviceptr cudaPtr = memoryManager->deviceAlloc(baseArray);
                base2Cuda[baseArray] = cudaPtr;
                baseArray->cudaPtr = cudaPtr;
                initCudaArray(baseArray);
            }
            operand->cudaPtr = baseArray->cudaPtr;
            op2Base[operand] = baseArray;
        }
    }
}

DataManagerSimple::DataManagerSimple(MemoryManager* memoryManager_) :
    memoryManager(memoryManager_),
    activeBatch(NULL) {}

void DataManagerSimple::lock(cphVBArray* operands[], 
                             int nops, 
                             InstructionBatch* batch)
{
    if (activeBatch == NULL)
    {
        activeBatch = batch;
    } 
    else if (batch != activeBatch)
    {
        activeBatch->execute();
        activeBatch = batch;
    }
    
    assert(nops > 0);
    mapOperands(operands, nops);
    
    cphVBArray* baseArray;
    /* We need to _sync all arrays that are read in the operation*/
    for (int i = 1; i < nops; ++i)
    {
        baseArray = op2Base[operands[i]];  
        _sync(baseArray);
    }
    /* Now we can just take the write lock on the array */
    baseArray = op2Base[operands[0]];
    writeLockTable[baseArray] = batch;
}

void DataManagerSimple::release(cphVBArray* baseArray)
{
#ifdef DEBUG
    std::cout << "DataManagerSimple::release()" << std::endl;
#endif
    sync(baseArray);
    discard(baseArray);
}

void DataManagerSimple::sync(cphVBArray* baseArray)
{
    assert(baseArray->base == NULL);
    _sync(baseArray);
    if (baseArray->data == NULL)
    {
        cphvb_data_ptr ptr = memoryManager->hostAlloc(baseArray);
        baseArray->data = ptr;
    }
    memoryManager->copyToHost(baseArray);
}

void DataManagerSimple::discard(cphVBArray* baseArray)
{
    assert(baseArray->base == NULL);
    memoryManager->free(baseArray);
    baseArray->cudaPtr = 0;
    base2Cuda.erase(baseArray);
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

void DataManagerSimple::flushAll()
{
    activeBatch->execute();
    activeBatch = NULL;
}



